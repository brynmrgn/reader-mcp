"""Thin async client for the Readeck REST API (read-only).

Paths/fields below are CONFIRMED against this Readeck instance (tested live), not
just the generic 0.20 docs. Differences from the library-sources original:
  - article is fetched as Markdown via /article.md (no HTML stripping needed)
  - listing is a bare JSON array with offset/limit paging (no `updated_since`
    param); incremental filtering is done by the caller against the `updated`
    field, since this instance was not confirmed to support server-side since-filtering
  - labels come back as [{name, count, href, ...}]
"""
from __future__ import annotations

import httpx

from . import config


class Readeck:
    def __init__(self, base_url: str = config.READECK_URL, token: str = config.READECK_TOKEN):
        self._base = base_url
        self._headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    async def _get(self, client: httpx.AsyncClient, path: str, **params):
        r = await client.get(f"{self._base}{path}", headers=self._headers, params=params or None)
        r.raise_for_status()
        return r

    async def bookmarks(self, limit: int = 50):
        """Yield bookmark dicts, newest-updated first, paging until exhausted.

        Confirmed: GET /api/bookmarks returns a bare JSON array; offset/limit paging
        works; each bookmark carries id, url, title, site_name, authors[], labels[],
        published, created, updated, state, has_article, is_deleted, description.
        Incremental behaviour is the caller's job (compare `updated` to its cursor).
        """
        async with httpx.AsyncClient(timeout=30) as client:
            offset = 0
            while True:
                r = await self._get(client, "/api/bookmarks",
                                    limit=limit, offset=offset, sort="-updated")
                items = r.json()
                if not items:
                    return
                for it in items:
                    yield it
                if len(items) < limit:
                    return
                offset += limit

    async def article_markdown(self, bookmark_id: str) -> str:
        """Extracted readable text as Markdown.

        Confirmed: GET /api/bookmarks/{id}/article.md returns Markdown (with YAML
        frontmatter). Already readable-extracted, so no tag stripping needed.
        """
        async with httpx.AsyncClient(timeout=60) as client:
            r = await self._get(client, f"/api/bookmarks/{bookmark_id}/article.md")
            return r.text

    async def labels(self) -> list[dict]:
        """Label taxonomy as [{name, count}].

        Confirmed: GET /api/bookmarks/labels returns [{name, count, href, ...}].
        Drops the extraction-failure bucket `_unextracted`.
        """
        async with httpx.AsyncClient(timeout=30) as client:
            r = await self._get(client, "/api/bookmarks/labels")
            data = r.json()
        out = []
        for item in data or []:
            name = item.get("name") if isinstance(item, dict) else item
            if name and name != "_unextracted":
                out.append({"name": name, "count": item.get("count") if isinstance(item, dict) else None})
        return out
