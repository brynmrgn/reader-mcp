"""Configuration for reader-mcp — all from the environment.

Standalone copy of the library-sources config, repointed at a separate Readeck
instance with its own sqlite-vec index. An empty required token/key means "not
configured" and the service refuses to start rather than serve an empty index.
"""
from __future__ import annotations

import os


def _req(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        raise RuntimeError(f"{name} is required (set it in the environment / .env)")
    return val


# Readeck instance + API token (Readeck UI -> profile -> API tokens). Read-only
# token is sufficient; the tools never write.
READECK_URL = os.environ.get("READECK_URL", "http://localhost:8091").rstrip("/")
READECK_TOKEN = os.environ.get("READECK_TOKEN", "")

# OpenAI embeddings. text-embedding-3-small: cheap, 1536-dim.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
EMBED_MODEL = os.environ.get("READER_EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = int(os.environ.get("READER_EMBED_DIM", "1536"))

# sqlite-vec index -- its OWN file, separate from library-sources.
DB_PATH = os.environ.get("READER_DB", "/home/ubuntu/reader-mcp/data/reader-index.sqlite")

# Chunking + sync cadence.
CHUNK_CHARS = int(os.environ.get("READER_CHUNK_CHARS", "1200"))
CHUNK_OVERLAP = int(os.environ.get("READER_CHUNK_OVERLAP", "150"))
SYNC_INTERVAL_SECS = int(os.environ.get("READER_SYNC_INTERVAL", "1200"))  # 20 min

# On first run the index is empty and every bookmark gets embedded. Cap how many
# bookmarks a single sync pass will process so the initial backfill is paced over
# several passes rather than one huge burst of OpenAI calls. Set 0 for no cap.
SYNC_MAX_PER_PASS = int(os.environ.get("READER_SYNC_MAX_PER_PASS", "200"))

# MCP HTTP surface: streamable-http at root "/", matching the estate convention so
# the connector uses the URL verbatim (no /mcp suffix). Bind localhost; the
# Cloudflare tunnel fronts TLS + Access.
HOST = os.environ.get("READER_MCP_HOST", "127.0.0.1")
PORT = int(os.environ.get("READER_MCP_PORT", "8095"))


def validate() -> None:
    """Fail fast at boot if the essentials are missing."""
    _req("READECK_TOKEN")
    _req("OPENAI_API_KEY")
