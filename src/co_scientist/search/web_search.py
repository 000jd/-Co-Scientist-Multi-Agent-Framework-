"""General web search for agentic research."""

import httpx
from typing import List, Dict, Any
from pydantic import BaseModel
from cachetools import TTLCache
import asyncio
import time


class WebSearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    source: str = "web"


class WebSearchEngine:
    """Agentic web search using Serper.dev (Google Search API) or similar."""
    
    def __init__(self, api_key: str, provider: str = "serper"):
        self.api_key = api_key
        self.provider = provider
        self._cache: TTLCache = TTLCache(maxsize=5000, ttl=3600)
        self._last_request = 0.0
        self._rate_limit = 1.0  # 1 req/sec for cheap tiers
        
    async def search(self, query: str, max_results: int = 10) -> List[WebSearchResult]:
        cache_key = f"web_{hash(query)}"
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        # Rate limit
        now = time.time()
        delay = max(0, self._rate_limit - (now - self._last_request))
        if delay > 0:
            await asyncio.sleep(delay)
            
        async with httpx.AsyncClient(timeout=30.0) as client:
            if self.provider == "serper":
                # Serper.dev - $50 for 5k searches, no card needed to start
                resp = await client.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": self.api_key},
                    json={"q": query, "num": max_results}
                )
                data = resp.json()
                results = []
                for item in data.get("organic", []):
                    results.append(WebSearchResult(
                        title=item.get("title", ""),
                        url=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                    ))
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
                
        self._last_request = time.time()
        self._cache[cache_key] = results
        return results
