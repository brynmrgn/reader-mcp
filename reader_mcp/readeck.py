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

    async def search_bookmarks(self, filters: dict, limit: int = 15) -> tuple[list[dict], int]:
        """Server-side bookmark search/filter via GET /api/bookmarks query params.

        Unlike bookmarks() (which pages everything for the sync loop), this passes the
        caller's filters straight to Readeck's own full-text + metadata search and returns
        a single bounded page plus the Total-Count. `filters` keys are Readeck param names
        (search, title, author, site, labels, type, is_marked, is_archived, read_status,
        range_start, range_end, sort, ...); None/empty values are dropped.

        Confirmed against this instance's OpenAPI: the list endpoint accepts these params
        and returns a bare JSON array with a `Total-Count` header (total across all pages).
        """
        params = {}
        for k, v in filters.items():
            if v in (None, "", []):
                continue
            # Readeck expects lowercase true/false, not Python's "True"/"False".
            params[k] = "true" if v is True else "false" if v is False else v
        params["limit"] = limit
        async with httpx.AsyncClient(timeout=30) as client:
            r = await self._get(client, "/api/bookmarks", **params)
            items = r.json() or []
            total = int(r.headers.get("Total-Count", len(items)))
        return items, total

    async def article_markdown(self, bookmark_id: str) -> str:
        """Extracted readable text as Markdown.

        Confirmed: GET /api/bookmarks/{id}/article.md returns Markdown (with YAML
        frontmatter). Already readable-extracted, so no tag stripping needed.
        """
        async with httpx.AsyncClient(timeout=60) as client:
            r = await self._get(client, f"/api/bookmarks/{bookmark_id}/article.md")
            return r.text

    async def annotations(self, bookmark_id: str) -> list[str]:
        """User highlights/annotations for one bookmark, as a list of text strings.

        Highlights are user-curated "this matters" passages, so they're indexed as their
        own high-signal chunks. Defensive: any non-200 (e.g. an instance without the
        endpoint) or shape drift yields [] so the bookmark still indexes from its body.

        VERIFY: endpoint path confirmed by probe against this instance
        (`/api/bookmarks/{id}/annotations`); item text under `text`.
        """
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await self._get(client, f"/api/bookmarks/{bookmark_id}/annotations")
                data = r.json()
        except Exception:  # noqa: BLE001
            return []
        items = data.get("items") if isinstance(data, dict) else data
        out = []
        for it in items or []:
            if isinstance(it, dict):
                # Readeck returns the highlighted passage under `text`; fall back to a
                # couple of plausible keys so a minor shape difference still captures it.
                t = it.get("text") or it.get("quote") or it.get("content") or it.get("body")
            else:
                t = it
            if t and str(t).strip():
                out.append(str(t).strip())
        return out

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
