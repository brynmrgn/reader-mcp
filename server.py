#!/usr/bin/env python3
"""Read-only MCP over a self-hosted Readeck instance.

Run over stdio; front it with supergateway for streamable HTTP, exactly like
the logseq-mcp bridge. Auth to Readeck is a bearer token (Settings > API Tokens).

  READECK_URL    e.g. http://localhost:8091   (talk to Readeck directly, not via Caddy)
  READECK_TOKEN  a read-scoped API token
"""

import os
from typing import Annotated

import httpx
from mcp.server.fastmcp import FastMCP

READECK_URL = os.environ["READECK_URL"].rstrip("/")
READECK_TOKEN = os.environ["READECK_TOKEN"]

# Semantic-search deps (optional: tools degrade to full-text if unset)
DATABASE_URL = os.environ.get("DATABASE_URL")
EMBED_URL = os.environ.get("EMBED_URL", "").rstrip("/")
EMBED_MODEL = os.environ.get("EMBED_MODEL")

mcp = FastMCP("reader-mcp")

_client = httpx.Client(
    base_url=f"{READECK_URL}/api",
    headers={"Authorization": f"Bearer {READECK_TOKEN}"},
    timeout=30.0,
)

_embed = httpx.Client(base_url=EMBED_URL, timeout=120.0) if EMBED_URL else None


def _embed_query(text: str) -> list[float]:
    r = _embed.post("/embeddings", json={"model": EMBED_MODEL, "input": [text]})
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


@mcp.tool()
def search_articles(
    query: Annotated[str, "Full-text search over saved articles. Empty string lists recent."],
    labels: Annotated[str, "Optional comma-separated labels to filter by."] = "",
    limit: Annotated[int, "Max results (1-50)."] = 10,
) -> list[dict]:
    """Search saved Readeck articles. Returns id, title, url, labels, and an excerpt."""
    params: dict[str, object] = {"limit": max(1, min(limit, 50))}
    if query:
        params["search"] = query
    if labels:
        params["labels"] = labels
    r = _client.get("/bookmarks", params=params)
    r.raise_for_status()
    items = r.json()
    # Readeck returns a list of bookmark objects; trim to what the model needs.
    return [
        {
            "id": b.get("id"),
            "title": b.get("title"),
            "url": b.get("url"),
            "labels": b.get("labels", []),
            "excerpt": (b.get("description") or "")[:300],
            "created": b.get("created"),
        }
        for b in items
    ]


@mcp.tool()
def get_article(
    id: Annotated[str, "Bookmark id from search_articles."],
) -> dict:
    """Fetch one article's readable content as Markdown, plus its metadata."""
    meta = _client.get(f"/bookmarks/{id}")
    meta.raise_for_status()
    md = _client.get(f"/bookmarks/{id}/article.md")  # confirm path on your instance
    md.raise_for_status()
    m = meta.json()
    return {
        "id": m.get("id"),
        "title": m.get("title"),
        "url": m.get("url"),
        "labels": m.get("labels", []),
        "content_markdown": md.text,
    }


@mcp.tool()
def list_labels() -> list[str]:
    """List all labels in use, so searches can be scoped sensibly."""
    r = _client.get("/bookmarks/labels")
    r.raise_for_status()
    data = r.json()
    # Readeck returns label objects; surface just the names.
    return [x.get("name", x) if isinstance(x, dict) else x for x in data]


@mcp.tool()
def semantic_search(
    query: Annotated[str, "Conceptual query; matches on meaning, not keywords."],
    labels: Annotated[str, "Optional comma-separated labels to scope the search."] = "",
    limit: Annotated[int, "Number of distinct articles to return (1-25)."] = 8,
) -> list[dict]:
    """Semantic search over indexed article content (pgvector). Returns the best
    articles with their closest-matching passage. Requires the indexer to have run."""
    if not (DATABASE_URL and _embed and EMBED_MODEL):
        return [{"error": "semantic search not configured; use search_articles instead"}]

    qvec = _embed_query(query)
    label_list = [l.strip() for l in labels.split(",") if l.strip()]

    # DISTINCT ON collapses to best passage per article; cosine distance ordering.
    sql = """
        SELECT DISTINCT ON (b.id)
               b.id, b.title, b.url, b.site_name, b.labels,
               c.text AS passage,
               (c.embedding <=> %s::vector) AS distance
        FROM reader_chunks c
        JOIN reader_bookmarks b ON b.id = c.bookmark_id
        {label_filter}
        ORDER BY b.id, distance
    """.format(label_filter="WHERE b.labels && %s" if label_list else "")

    params: list = [qvec]
    if label_list:
        params.append(label_list)

    import psycopg
    with psycopg.connect(DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    rows.sort(key=lambda r: r[6])  # by distance across the DISTINCT-ON set
    return [
        {
            "id": r[0], "title": r[1], "url": r[2], "site_name": r[3],
            "labels": r[4], "passage": r[5][:500],
            "score": round(1 - r[6], 3),  # cosine similarity
        }
        for r in rows[: max(1, min(limit, 25))]
    ]


if __name__ == "__main__":
    mcp.run()  # stdio transport
