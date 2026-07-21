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
    db.commit()
    db.close()


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


async def sync_once(readeck: Readeck) -> int:
    """Pull bookmarks whose `updated` is newer than our cursor, (re)embed and index.
    Only article-bearing, loaded, non-deleted bookmarks. Paced by SYNC_MAX_PER_PASS.
    Returns the number of bookmarks processed this pass."""
    db = _connect()
    row = db.execute("SELECT v FROM sync_state WHERE k='updated_since'").fetchone()
    since = row["v"] if row else None
    processed, newest = 0, since
    try:
        async for bm in readeck.bookmarks():
            upd = bm.get("updated")
            # Listing is newest-updated first; once we reach items at/older than the
            # cursor, everything after is already indexed -> stop early.
            if since and upd and upd <= since:
                break
            if not (bm.get("state") == 0 and bm.get("has_article") and not bm.get("is_deleted")):
                continue
            text = await readeck.article_markdown(str(bm.get("id")))
            if not text:
                continue
            chunks = _chunk(text)
            vectors = await _embed(chunks)
            _replace_bookmark(db, bm, chunks, vectors)
            db.commit()
            processed += 1
            if upd and (newest is None or upd > newest):
                newest = upd
            if config.SYNC_MAX_PER_PASS and processed >= config.SYNC_MAX_PER_PASS:
                break
        if newest and newest != since:
            db.execute("INSERT OR REPLACE INTO sync_state(k, v) VALUES ('updated_since', ?)",
                       (newest,))
            db.commit()
    finally:
        db.close()
    return processed


async def sync_loop() -> None:
    """Boot sync, then every SYNC_INTERVAL_SECS. Errors logged and retried next tick
    so a Readeck/OpenAI blip never takes the query path down."""
    readeck = Readeck()
    while True:
        try:
            n = await sync_once(readeck)
            print(f"[reader-mcp] sync: {n} bookmarks indexed", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[reader-mcp] sync failed: {type(e).__name__}: {e}", flush=True)
        await asyncio.sleep(config.SYNC_INTERVAL_SECS)


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
