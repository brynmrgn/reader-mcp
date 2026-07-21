"""Semantic index: sqlite-vec store + OpenAI embeddings + sync loop.

Standalone copy of the library-sources index, adapted for this Readeck instance:
article text comes from /article.md, incremental sync filters on the `updated`
field client-side, and only loaded article-bearing bookmarks are indexed.
"""
from __future__ import annotations

import asyncio
import json
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
        "  text TEXT, title TEXT, url TEXT, site_name TEXT, labels TEXT, date TEXT)"
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


def _migrate(db: sqlite3.Connection) -> None:
    """One-time move to the paced-backfill scheme (schema v2).

    The previous scheme advanced the high-water cursor to the newest bookmark after a
    *capped* pass, which froze the backfill at SYNC_MAX_PER_PASS: every later pass
    early-broke on the first (newest) item. Seed indexed_state from whatever chunks
    already exist (so we don't re-embed them) and drop that premature cursor so the
    backfill resumes from where it stalled.
    """
    ver = db.execute("SELECT v FROM sync_state WHERE k='schema_version'").fetchone()
    if ver and int(ver["v"]) >= 2:
        return
    db.execute("INSERT OR IGNORE INTO indexed_state(bookmark_id, updated) "
               "SELECT DISTINCT bookmark_id, NULL FROM chunks")
    db.execute("DELETE FROM sync_state WHERE k='updated_since'")
    db.execute("INSERT OR REPLACE INTO sync_state(k, v) VALUES ('schema_version', '2')")


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _chunk(text: str) -> list[str]:
    step = max(1, config.CHUNK_CHARS - config.CHUNK_OVERLAP)
    return [text[i:i + config.CHUNK_CHARS] for i in range(0, len(text), step)] or [""]


async def _embed(texts: list[str]) -> list[list[float]]:
    # Client constructed inside the call so its httpx session binds to whichever
    # event loop is running (server loop for queries, sync-thread loop for indexing)
    # -- sharing one client across both loops breaks httpx.
    async with AsyncOpenAI(api_key=config.OPENAI_API_KEY) as client:
        resp = await client.embeddings.create(model=config.EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


# -- write path (sync) -----------------------------------------------------------

def _replace_bookmark(db: sqlite3.Connection, bm: dict, chunks: list[str],
                      vectors: list[list[float]]) -> None:
    bid = str(bm.get("id"))
    old = [r["rowid"] for r in db.execute("SELECT rowid FROM chunks WHERE bookmark_id=?", (bid,))]
    for rid in old:
        db.execute("DELETE FROM vec_chunks WHERE rowid=?", (rid,))
    db.execute("DELETE FROM chunks WHERE bookmark_id=?", (bid,))
    labels = json.dumps(bm.get("labels") or [])
    for idx, (text, vec) in enumerate(zip(chunks, vectors)):
        cur = db.execute(
            "INSERT INTO chunks(bookmark_id, chunk_idx, text, title, url, site_name, labels, date)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (bid, idx, text, bm.get("title"), bm.get("url"), bm.get("site_name"),
             labels, bm.get("published") or bm.get("created")),
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
            chunks = _chunk(text)
            vectors = await _embed(chunks)
            _replace_bookmark(db, bm, chunks, vectors)
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

async def search(q: str, limit: int = 8) -> list[dict]:
    """Semantic search: embed query, KNN over chunk vectors, collapse to distinct
    bookmarks (best chunk wins)."""
    (qvec,) = await _embed([q])
    db = _connect()
    try:
        rows = db.execute(
            "WITH knn AS ("
            "  SELECT rowid, distance FROM vec_chunks"
            "  WHERE embedding MATCH ? ORDER BY distance LIMIT ?"
            ")"
            " SELECT c.bookmark_id, c.text, c.title, c.url, c.site_name, c.labels, c.date, knn.distance"
            " FROM knn JOIN chunks c ON c.rowid = knn.rowid"
            " ORDER BY knn.distance",
            (_pack(qvec), max(limit * 4, limit)),
        ).fetchall()
    finally:
        db.close()
    out, seen = [], set()
    for r in rows:
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
            "labels": json.loads(r["labels"] or "[]"),
            "date": r["date"],
            "score": round(1 - r["distance"], 4),
        })
        if len(out) >= limit:
            break
    return out


async def get_source(bookmark_id: str) -> dict:
    """Full readable text (all chunks in order) + metadata for one bookmark."""
    db = _connect()
    try:
        rows = db.execute(
            "SELECT text, title, url, site_name, labels, date FROM chunks"
            " WHERE bookmark_id=? ORDER BY chunk_idx", (str(bookmark_id),),
        ).fetchall()
    finally:
        db.close()
    if not rows:
        return {}
    return {
        "id": str(bookmark_id),
        "uri": rows[0]["url"], "title": rows[0]["title"],
        "site_name": rows[0]["site_name"],
        "labels": json.loads(rows[0]["labels"] or "[]"), "date": rows[0]["date"],
        "text": " ".join(r["text"] for r in rows),
    }
