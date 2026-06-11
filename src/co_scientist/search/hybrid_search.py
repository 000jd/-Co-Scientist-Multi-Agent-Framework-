"""
DuckDuckGo Search + Tinyfish Fetch API
"""

import asyncio
import re
from typing import Any, List
import httpx

from co_scientist.search.web_search import WebSearchResult

MAX_CONTENT_CHARS = 4000
TINYFISH_FETCH_URL = "https://api.fetch.tinyfish.ai"


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _ddg_search(
    query: str,
    max_results: int = 5,
) -> list[dict[str, str]]:

    results: list[dict[str, str]] = []

    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            for item in ddgs.text(
                query,
                max_results=max_results,
            ):
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("href", ""),
                        "snippet": item.get("body", ""),
                    }
                )
    except Exception as e:
        import logging
        logging.getLogger("hybrid_search").warning(f"DDGS search failed for query={query}: {e}")

    return results


async def _fetch_page_tinyfish(
    client: httpx.AsyncClient,
    url: str,
    api_key: str
) -> str:

    try:
        response = await client.post(
            TINYFISH_FETCH_URL,
            json={
                "urls": [url],
                "format": "markdown",
            },
            headers={
                "X-API-Key": api_key,
                "Content-Type": "application/json",
            },
        )

        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])

        if not results:
            return ""

        text = results[0].get("text", "")

        if not isinstance(text, str):
            text = str(text)

        return _clean_text(text)[:MAX_CONTENT_CHARS]

    except Exception as e:
        return ""


class HybridSearchEngine:
    """Agentic web search using DuckDuckGo + Tinyfish."""
    
    def __init__(self, api_key: str):
        self.tinyfish_api_key = api_key
        
    async def search_and_fetch(self, query: str, max_results: int = 5) -> List[WebSearchResult]:
        hits = await asyncio.to_thread(
            _ddg_search,
            query,
            max_results,
        )

        async with httpx.AsyncClient(
            timeout=60,
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
            ),
        ) as client:
            contents = await asyncio.gather(
                *[
                    _fetch_page_tinyfish(
                        client,
                        hit["url"],
                        self.tinyfish_api_key
                    )
                    for hit in hits
                ]
            )

        final_results = []
        for hit, content in zip(hits, contents):
            final_results.append(
                WebSearchResult(
                    title=hit["title"],
                    url=hit["url"],
                    snippet=content if content else hit.get("snippet", ""),
                )
            )
            
        return final_results
