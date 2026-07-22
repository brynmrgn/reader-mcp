"""reader-mcp: three read-only tools over the semantic index of a personal Readeck
instance, plus the background sync loop. Streamable-HTTP at root, native FastMCP 2.x
(3.x breaks the Claude connector), default transport behaviour -- matching the
proven library-sources pattern.
"""
from __future__ import annotations

import asyncio
import threading

from fastmcp import FastMCP

from . import config, index
from .readeck import Readeck

mcp = FastMCP("reader-mcp")


@mcp.tool()
async def search_articles(query: str, limit: int = 8, label: str | None = None,
                          site_name: str | None = None,
                          published_after: str | None = None, published_before: str | None = None,
                          added_after: str | None = None, added_before: str | None = None,
                          updated_after: str | None = None, updated_before: str | None = None) -> list[dict]:
    """Semantic search over saved articles in the personal reading library. Returns the
    most relevant passages, each with the original `uri`, `title`, `site_name`, `labels`,
    `published` / `added` / `updated` dates, a relevance `score`, and the `kind` of passage
    that matched (`body`, `summary`, or a user `highlight`) plus its `section`. Use `query`
    in natural language.

    Optional filters:
      * `label` — scope to one label from list_labels (e.g. 'parliament').
      * `site_name` — restrict to one source (e.g. 'The Guardian'), case-insensitive.
      * date ranges — each dimension takes an inclusive `_after` and exclusive `_before`
        ISO date/datetime bound:
          - `published_after` / `published_before` — the article's publish date.
          - `added_after` / `added_before` — when it was saved to the library ('what did
            I save this week').
          - `updated_after` / `updated_before` — last modified in Readeck.
    User highlights and summaries are given a small ranking boost. (To LIST articles by
    date with no topic, or for exact keyword lookup, use find_articles instead.)"""
    return await index.search(query, limit=limit, label=label, site_name=site_name,
                              published_after=published_after, published_before=published_before,
                              added_after=added_after, added_before=added_before,
                              updated_after=updated_after, updated_before=updated_before)


def _fmt_bookmark(bm: dict) -> dict:
    """Shape a raw Readeck bookmark into the fields a caller needs to triage results."""
    return {
        "id": bm.get("id"),
        "uri": bm.get("url"),
        "title": bm.get("title"),
        "site_name": bm.get("site_name"),
        "authors": bm.get("authors") or [],
        "labels": bm.get("labels") or [],
        "published": bm.get("published"),   # article publish date (may be null)
        "added": bm.get("created"),         # when it was saved to the library
        "updated": bm.get("updated"),       # last modified in Readeck
        "type": bm.get("type"),
        "reading_time": bm.get("reading_time"),
        "read_progress": bm.get("read_progress"),
        "is_marked": bm.get("is_marked"),
        "is_archived": bm.get("is_archived"),
        "description": bm.get("description"),
    }


@mcp.tool()
async def find_articles(query: str | None = None, title: str | None = None,
                        author: str | None = None, site: str | None = None,
                        label: str | None = None, added_after: str | None = None,
                        added_before: str | None = None, read_status: str | None = None,
                        is_favorite: bool | None = None, is_archived: bool | None = None,
                        sort: str = "-added", limit: int = 15) -> dict:
    """Exact keyword + metadata search over the reading library, straight from Readeck
    (no semantic ranking). Use this — NOT search_articles — when the request is about
    *which* articles match precise criteria rather than their meaning: exact words/phrases,
    a title/author/site, a label, favorites/archive, or WHEN something was saved. It
    returns a complete, deterministically-ordered list with a `total` count, and needs no
    `query` at all — so it's the right tool for 'what did I add yesterday / this week'.

    Filters (all optional, combined with AND):
      * `query` — full-text search over title + content (exact keywords, not meaning).
      * `title` / `author` / `site` — match those fields.
      * `label` — one or more labels (comma-separated), from list_labels.
      * `added_after` / `added_before` — ISO date (e.g. '2026-07-21'); filters on the date
        the article was ADDED (saved) to the library. For 'yesterday' pass
        added_after=<yesterday's date>. (Publish/updated date ranges: use search_articles.)
      * `read_status` — 'unread', 'reading', or 'read'.
      * `is_favorite` / `is_archived` — booleans.
      * `sort` — default '-added' (newest saved first); also 'added', '-published' /
        'published', 'title', 'site'.
      * `limit` — max results (default 15).

    For conceptual/meaning-based search, use search_articles instead."""
    sort_map = {"-added": "-created", "added": "created",
                "-published": "-published", "published": "published",
                "title": "title", "site": "site"}
    items, total = await Readeck().search_bookmarks({
        "search": query, "title": title, "author": author, "site": site,
        "labels": label, "range_start": added_after, "range_end": added_before,
        "read_status": read_status,
        "is_marked": is_favorite, "is_archived": is_archived,
        "sort": sort_map.get(sort, "-created"),
    }, limit=limit)
    return {"total": total, "count": len(items),
            "articles": [_fmt_bookmark(bm) for bm in items]}


@mcp.tool()
async def get_article(id: str) -> dict:
    """Read one saved article in full (all text + metadata) by its id -- use after
    search_articles when you need the surrounding detail or full argument."""
    return await index.get_source(id)


@mcp.tool()
async def list_labels() -> list[dict]:
    """List the labels in the reading library (with counts), so a search can be
    scoped to a relevant set (e.g. parliament, music, software_dev)."""
    return await Readeck().labels()


def _start_sync_thread() -> None:
    """Run the sync loop in its own daemon thread with its own event loop, so it
    indexes in the background independently of FastMCP's request loop. Each embed
    call builds a loop-local OpenAI client, so the two loops don't share httpx."""
    threading.Thread(target=lambda: asyncio.run(index.sync_loop()),
                     name="reader-sync", daemon=True).start()


def main() -> None:
    config.validate()
    index.init_db()
    _start_sync_thread()
    mcp.run(transport="streamable-http", host=config.HOST, port=config.PORT, path="/mcp")


if __name__ == "__main__":
    main()
