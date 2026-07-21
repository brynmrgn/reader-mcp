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
async def search_articles(query: str, limit: int = 8) -> list[dict]:
    """Semantic search over saved articles in the personal reading library. Returns
    the most relevant passages, each with the original `uri`, `title`, `site_name`,
    `labels`, `date`, and a relevance `score`. Use `query` in natural language."""
    return await index.search(query, limit=limit)


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
