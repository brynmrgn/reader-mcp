"""Semantic index: sqlite-vec store + OpenAI embeddings + sync loop.

Standalone copy of the library-sources index, adapted for this Readeck instance:
article text comes from /article.md, incremental sync filters on the `updated`
field client-side, and only loaded article-bearing bookmarks are indexed.
"""
from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import struct

import sqlite_vec
from openai import AsyncOpenAI

from . import config
from .readeck import Readeck


def _connect() -> sqlite3.Connection:
    db = sqlite3.connect(config.DB_PATH)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.row_factory = sqlite3.Row
    return db


def init_db() -> None:
    import os
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    db = _connect()
    db.execute(
        "CREATE TABLE IF NOT EXISTS chunks ("
        "  rowid INTEGER PRIMARY KEY, bookmark_id TEXT, chunk_idx INTEGER,"
        "  text TEXT, kind TEXT, section TEXT,"
        "  title TEXT, url TEXT, site_name TEXT, labels TEXT, date TEXT)"
    )
    db.execute("CREATE INDEX IF NOT EXISTS chunks_bm ON chunks(bookmark_id)")
    db.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
        f"  embedding float[{config.EMBED_DIM}])"
    )
    db.execute("CREATE TABLE IF NOT EXISTS sync_state (k TEXT PRIMARY KEY, v TEXT)")
    # Per-bookmark index of the `updated` value we last embedded, so a paced backfill
    # can walk newest→oldest across passes, skipping what's already current instead of
    # re-embedding the newest N every pass. NULL = a legacy row (indexed before this
    # table existed): treated as current so the backfill still moves forward.
    db.execute("CREATE TABLE IF NOT EXISTS indexed_state ("
               "  bookmark_id TEXT PRIMARY KEY, updated TEXT)")
    _migrate(db)
    db.commit()
    db.close()


_SCHEMA_VERSION = 3


def _migrate(db: sqlite3.Connection) -> None:
    """Run pending schema migrations in order.

    v2 — paced-backfill scheme. The old code advanced the high-water cursor to the
    newest bookmark after a *capped* pass, freezing the backfill at SYNC_MAX_PER_PASS
    (every later pass early-broke on the first item). Seed indexed_state from existing
    chunks (so we don't re-embed them) and drop the premature cursor.

    v3 — structure-aware chunking + summary/highlight parts. Chunk boundaries and the
    per-chunk `kind`/`section` are new, so the old fixed-window chunks are stale: add
    the columns and wipe the index to force a full re-embed (the paced backfill drains
    it over the next passes).
    """
    row = db.execute("SELECT v FROM sync_state WHERE k='schema_version'").fetchone()
    ver = int(row["v"]) if row else 0
    if ver >= _SCHEMA_VERSION:
        return
    if ver < 2:
        db.execute("INSERT OR IGNORE INTO indexed_state(bookmark_id, updated) "
                   "SELECT DISTINCT bookmark_id, NULL FROM chunks")
        db.execute("DELETE FROM sync_state WHERE k='updated_since'")
    if ver < 3:
        cols = {r[1] for r in db.execute("PRAGMA table_info(chunks)")}
        if "kind" not in cols:
            db.execute("ALTER TABLE chunks ADD COLUMN kind TEXT")
        if "section" not in cols:
            db.execute("ALTER TABLE chunks ADD COLUMN section TEXT")
        db.execute("DELETE FROM vec_chunks")
        db.execute("DELETE FROM chunks")
        db.execute("DELETE FROM indexed_state")
        db.execute("DELETE FROM sync_state WHERE k='updated_since'")
        print("[reader-mcp] migration v3: re-indexing all bookmarks with "
              "structure-aware chunking (summary + highlights + sections)", flush=True)
    db.execute("INSERT OR REPLACE INTO sync_state(k, v) VALUES ('schema_version', ?)",
               (str(_SCHEMA_VERSION),))


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


# -- structure-aware chunking ----------------------------------------------------
# A bookmark becomes a list of "parts", each {text, kind, section}:
#   * summary   — the bookmark's description/abstract (one part, if present)
#   * highlight — each user highlight/annotation (high-signal, user-curated)
#   * body      — the article split by Markdown section, then packed by paragraph
# `text` is stored/displayed verbatim; the section breadcrumb is prepended to the
# EMBED input only (see _embed_input), so vectors get heading context without the
# stored/rejoined text repeating headings.

_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.S)   # article.md YAML block
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_PARA_SPLIT_RE = re.compile(r"\n\s*\n")


def _strip_frontmatter(md: str) -> str:
    return _FRONTMATTER_RE.sub("", md, count=1)


def _split_sections(md: str) -> list[tuple[str, str]]:
    """Split Markdown into (section_path, text) on ATX headings. section_path is the
    breadcrumb of enclosing headings, e.g. 'Results > Cohort A'. Falls back to a single
    ('', whole-text) section when there are no headings."""
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append((" > ".join(t for _, t in stack), body))

    for line in md.split("\n"):
        m = _HEADING_RE.match(line)
        if m:
            flush()
            buf.clear()
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buf.append(line)
    flush()
    return sections or [("", md.strip())]


def _pack_paragraphs(text: str, max_chars: int, overlap: int) -> list[str]:
    """Greedily pack whole paragraphs into <=max_chars chunks (never splitting a
    paragraph unless it alone exceeds the limit, in which case it's hard-split with a
    char overlap so nothing is dropped)."""
    chunks: list[str] = []
    cur = ""
    for para in (p.strip() for p in _PARA_SPLIT_RE.split(text) if p.strip()):
        if len(para) > max_chars:
            if cur:
                chunks.append(cur)
                cur = ""
            step = max(1, max_chars - overlap)
            chunks.extend(para[i:i + max_chars] for i in range(0, len(para), step))
        elif cur and len(cur) + 2 + len(para) > max_chars:
            chunks.append(cur)
            cur = para
        else:
            cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        chunks.append(cur)
    return chunks


def _build_parts(bm: dict, markdown: str, highlights: list[str]) -> list[dict]:
    """Assemble a bookmark's indexable parts: summary, highlights, then body sections."""
    parts: list[dict] = []
    summary = (bm.get("description") or "").strip()
    if summary:
        parts.append({"text": summary, "kind": "summary", "section": None})
    for h in highlights or []:
        h = (h or "").strip()
        if h:
            parts.append({"text": h, "kind": "highlight", "section": None})
    body = _strip_frontmatter(markdown or "")
    for section, sect_text in _split_sections(body):
        for piece in _pack_paragraphs(sect_text, config.CHUNK_CHARS, config.CHUNK_OVERLAP):
            parts.append({"text": piece, "kind": "body", "section": section or None})
    if not parts:
        parts.append({"text": (markdown or "").strip(), "kind": "body", "section": None})
    return parts


def _embed_input(part: dict) -> str:
    """The string actually embedded: body chunks get their section breadcrumb prepended
    for context; summary/highlight parts embed verbatim."""
    if part.get("section"):
        return f"{part['section']}\n\n{part['text']}"
    return part["text"]


async def _embed(texts: list[str]) -> list[list[float]]:
    # Client constructed inside the call so its httpx session binds to whichever
    # event loop is running (server loop for queries, sync-thread loop for indexing)
    # -- sharing one client across both loops breaks httpx.
    async with AsyncOpenAI(api_key=config.OPENAI_API_KEY) as client:
        resp = await client.embeddings.create(model=config.EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


# -- write path (sync) -----------------------------------------------------------

def _replace_bookmark(db: sqlite3.Connection, bm: dict, parts: list[dict],
                      vectors: list[list[float]]) -> None:
    bid = str(bm.get("id"))
    old = [r["rowid"] for r in db.execute("SELECT rowid FROM chunks WHERE bookmark_id=?", (bid,))]
    for rid in old:
        db.execute("DELETE FROM vec_chunks WHERE rowid=?", (rid,))
    db.execute("DELETE FROM chunks WHERE bookmark_id=?", (bid,))
    labels = json.dumps(bm.get("labels") or [])
    date = bm.get("published") or bm.get("created")
    for idx, (part, vec) in enumerate(zip(parts, vectors)):
        cur = db.execute(
            "INSERT INTO chunks(bookmark_id, chunk_idx, text, kind, section,"
            " title, url, site_name, labels, date) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (bid, idx, part["text"], part["kind"], part.get("section"),
             bm.get("title"), bm.get("url"), bm.get("site_name"), labels, date),
        )
        db.execute("INSERT INTO vec_chunks(rowid, embedding) VALUES (?,?)",
                   (cur.lastrowid, _pack(vec)))


async def sync_once(readeck: Readeck) -> tuple[int, bool]:
    """Index up to SYNC_MAX_PER_PASS not-yet-current bookmarks, newest→oldest.

    Only article-bearing, loaded, non-deleted bookmarks. Returns (processed, capped),
    where `capped` means the pass stopped at the cap and there is more backlog to do.

    Two cursors work together so the cap and incremental catch-up don't fight:
      * `indexed_state[bid]` — the `updated` we last embedded per bookmark. A pass
        walks newest→oldest and SKIPS anything already current, so successive capped
        passes march down the backlog instead of re-embedding the newest N each time.
      * `updated_since` — a high-water mark, advanced ONLY after a non-capped pass
        (i.e. the whole backlog above it is indexed). Then later passes can early-break
        once they reach items at/older than it. It stays unset while backfilling.
    """
    db = _connect()
    row = db.execute("SELECT v FROM sync_state WHERE k='updated_since'").fetchone()
    since = row["v"] if row else None
    indexed = dict(db.execute("SELECT bookmark_id, updated FROM indexed_state").fetchall())
    processed = skipped = scanned = 0
    newest, capped = since, False
    try:
        async for bm in readeck.bookmarks():
            scanned += 1
            upd = bm.get("updated")
            if upd and (newest is None or upd > newest):
                newest = upd
            # Steady state: a cursor exists only after a non-capped pass, so everything
            # at/older than it is fully indexed -> stop (listing is newest-first).
            if since and upd and upd <= since:
                break
            if not (bm.get("state") == 0 and bm.get("has_article") and not bm.get("is_deleted")):
                continue
            bid = str(bm.get("id"))
            # Already current? (NULL = legacy row, treat as current so backfill advances.)
            if bid in indexed and (indexed[bid] is None or indexed[bid] == upd):
                skipped += 1
                continue
            text = await readeck.article_markdown(bid)
            if not text:
                continue
            highlights = await readeck.annotations(bid)
            parts = _build_parts(bm, text, highlights)
            vectors = await _embed([_embed_input(p) for p in parts])
            _replace_bookmark(db, bm, parts, vectors)
            db.execute("INSERT OR REPLACE INTO indexed_state(bookmark_id, updated) VALUES (?,?)",
                       (bid, upd))
            db.commit()
            indexed[bid] = upd
            processed += 1
            if config.SYNC_MAX_PER_PASS and processed >= config.SYNC_MAX_PER_PASS:
                capped = True
                break
        # Only claim the high-water mark once a pass finished WITHOUT hitting the cap:
        # then the whole backlog above `newest` really is indexed. While capped, leave
        # the cursor so the next pass walks from the top, skips the done ones, continues.
        if not capped and newest and newest != since:
            db.execute("INSERT OR REPLACE INTO sync_state(k, v) VALUES ('updated_since', ?)",
                       (newest,))
            db.commit()
    finally:
        db.close()
    print(f"[reader-mcp] sync: {processed} indexed, {skipped} up-to-date, {scanned} scanned"
          + (" (capped — more backlog)" if capped else ""), flush=True)
    return processed, capped


async def sync_loop() -> None:
    """Boot sync, then repeat. While a pass is capped there's backlog left, so loop
    again after a short pause to drain the initial backfill quickly; once caught up,
    settle into SYNC_INTERVAL_SECS. Errors are logged and retried so a Readeck/OpenAI
    blip never takes the query path down."""
    readeck = Readeck()
    while True:
        capped = False
        try:
            _, capped = await sync_once(readeck)
        except Exception as e:  # noqa: BLE001
            print(f"[reader-mcp] sync failed: {type(e).__name__}: {e}", flush=True)
        await asyncio.sleep(5 if capped else config.SYNC_INTERVAL_SECS)


# -- read path (tools) -----------------------------------------------------------

# User-curated / abstractive parts are stronger relevance signals than raw body text,
# so nudge them up in the ranking (heuristic, on the 1-distance similarity proxy).
_KIND_BOOST = {"highlight": 0.06, "summary": 0.03}


async def search(q: str, limit: int = 8, label: str | None = None,
                 site_name: str | None = None, since: str | None = None) -> list[dict]:
    """Semantic search: embed query, KNN over chunk vectors, apply optional metadata
    filters, boost highlight/summary chunks, then collapse to distinct bookmarks (best
    chunk wins). Filters can't go inside the vec0 KNN (it needs the LIMIT directly), so
    we over-fetch a larger candidate set and filter/rank in Python."""
    (qvec,) = await _embed([q])
    filtered = bool(label or site_name or since)
    k = max(limit * 25, 300) if filtered else max(limit * 4, limit)
    db = _connect()
    try:
        rows = db.execute(
            "WITH knn AS ("
            "  SELECT rowid, distance FROM vec_chunks"
            "  WHERE embedding MATCH ? ORDER BY distance LIMIT ?"
            ")"
            " SELECT c.bookmark_id, c.text, c.kind, c.section, c.title, c.url,"
            " c.site_name, c.labels, c.date, knn.distance"
            " FROM knn JOIN chunks c ON c.rowid = knn.rowid"
            " ORDER BY knn.distance",
            (_pack(qvec), k),
        ).fetchall()
    finally:
        db.close()
    site_l = site_name.lower() if site_name else None
    scored = []
    for r in rows:
        lbls = json.loads(r["labels"] or "[]")
        if label and label not in lbls:
            continue
        if site_l and (r["site_name"] or "").lower() != site_l:
            continue
        if since and (not r["date"] or r["date"] < since):
            continue
        score = (1 - r["distance"]) + _KIND_BOOST.get(r["kind"], 0.0)
        scored.append((score, r, lbls))
    scored.sort(key=lambda t: t[0], reverse=True)
    out, seen = [], set()
    for score, r, lbls in scored:
        bid = r["bookmark_id"]
        if bid in seen:
            continue
        seen.add(bid)
        out.append({
            "id": bid,
            "uri": r["url"],
            "title": r["title"],
            "site_name": r["site_name"],
            "text": r["text"],
            "kind": r["kind"],
            "section": r["section"],
            "labels": lbls,
            "date": r["date"],
            "score": round(score, 4),
        })
        if len(out) >= limit:
            break
    return out


async def get_source(bookmark_id: str) -> dict:
    """Full readable text + metadata for one bookmark, with the summary and user
    highlights surfaced separately from the body."""
    db = _connect()
    try:
        rows = db.execute(
            "SELECT text, kind, section, title, url, site_name, labels, date FROM chunks"
            " WHERE bookmark_id=? ORDER BY chunk_idx", (str(bookmark_id),),
        ).fetchall()
    finally:
        db.close()
    if not rows:
        return {}
    summary = next((r["text"] for r in rows if r["kind"] == "summary"), None)
    highlights = [r["text"] for r in rows if r["kind"] == "highlight"]
    body = " ".join(r["text"] for r in rows if r["kind"] == "body")
    return {
        "id": str(bookmark_id),
        "uri": rows[0]["url"], "title": rows[0]["title"],
        "site_name": rows[0]["site_name"],
        "labels": json.loads(rows[0]["labels"] or "[]"), "date": rows[0]["date"],
        "summary": summary,
        "highlights": highlights,
        "text": body or " ".join(r["text"] for r in rows),
    }
