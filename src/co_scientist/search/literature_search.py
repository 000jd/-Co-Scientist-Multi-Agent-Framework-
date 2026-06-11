"""
Literature Search Infrastructure for Co-Scientist.

Multi-source search across PubMed, arXiv, Semantic Scholar, CrossRef, and Google Scholar.
Handles rate limiting, caching, and result deduplication.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any, Dict, List, Optional
from urllib.parse import quote
import structlog
import httpx
from cachetools import TTLCache

logger = structlog.get_logger(__name__)


class LiteratureSearcher:
    """Multi-source literature search with caching and rate limiting."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Max 10,000 queries, TTL of 24 hours
        self._cache: TTLCache = TTLCache(maxsize=10000, ttl=86400)
        self._last_request_time: Dict[str, float] = {}
        self._rate_limit = config.get("rate_limit_per_second", 2.0)
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                limits=httpx.Limits(max_connections=10),
                headers={"User-Agent": "CoScientist/1.0 (Research Framework)"},
                follow_redirects=True
            )
        return self._client

    async def search(
        self,
        query: str,
        max_results: int = 50,
        sources: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search across multiple sources."""
        sources = sources or ["semantic_scholar", "crossref", "eartharxiv"]
        all_results = []

        tasks = []
        for source in sources:
            if source == "pubmed":
                tasks.append(self._search_pubmed(query, max_results))
            elif source == "semantic_scholar":
                tasks.append(self._search_semantic_scholar(query, max_results))
            elif source == "arxiv":
                tasks.append(self._search_arxiv(query, max_results))
            elif source == "crossref":
                tasks.append(self._search_crossref(query, max_results))
            elif source == "eartharxiv":
                tasks.append(self._search_eartharxiv(query, max_results))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_results.extend(result)
            elif isinstance(result, Exception):
                logger.warning("search.source_failed", error=str(result))

        # Deduplicate by title
        seen_titles = set()
        deduped = []
        for r in all_results:
            title = r.get("title", "").lower().strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                deduped.append(r)

        # Sort by relevance (year + citation count proxy)
        deduped.sort(
            key=lambda x: (x.get("year", 0), x.get("citation_count", 0)),
            reverse=True,
        )

        return deduped[:max_results]

    async def _search_pubmed(
        self,
        query: str,
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """Search PubMed via E-utilities."""
        try:
            cache_key = f"pubmed_{hashlib.md5(query.encode()).hexdigest()}"
            if cache_key in self._cache:
                return self._cache[cache_key]

            await self._rate_limit_delay("pubmed")

            client = await self._get_client()

            # Search for IDs
            search_url = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                f"?db=pubmed&term={quote(query)}&retmax={min(max_results, 100)}"
                f"&retmode=json"
            )

            resp = await client.get(search_url)
            data = resp.json()
            ids = data.get("esearchresult", {}).get("idlist", [])

            if not ids:
                return []

            # Fetch summaries
            summary_url = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                f"?db=pubmed&id={','.join(ids[:20])}&retmode=json"
            )

            resp = await client.get(summary_url)
            data = resp.json()

            results = []
            for uid, doc in data.get("result", {}).items():
                if uid == "uids":
                    continue
                results.append({
                    "title": doc.get("title", ""),
                    "authors": ", ".join(a.get("name", "") for a in doc.get("authors", [])[:5]),
                    "year": doc.get("pubdate", "")[:4],
                    "journal": doc.get("fulljournalname", ""),
                    "doi": doc.get("elocationid", ""),
                    "abstract": doc.get("title", ""),
                    "source": "pubmed",
                    "pmid": uid,
                })

            self._cache[cache_key] = results
            return results

        except Exception as e:
            logger.warning("search.pubmed_failed", error=str(e))
            return []

    async def _search_semantic_scholar(
        self,
        query: str,
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """Search Semantic Scholar."""
        try:
            cache_key = f"ss_{hashlib.md5(query.encode()).hexdigest()}"
            if cache_key in self._cache:
                return self._cache[cache_key]

            await self._rate_limit_delay("semantic_scholar")

            client = await self._get_client()

            url = (
                f"https://api.semanticscholar.org/graph/v1/paper/search"
                f"?query={quote(query)}&limit={min(max_results, 100)}"
                f"&fields=title,authors,year,abstract,citationCount,venue"
            )

            headers = {}
            api_key = self.config.get("semantic_scholar_api_key")
            if api_key:
                headers["x-api-key"] = api_key

            resp = await client.get(url, headers=headers)
            data = resp.json()

            results = []
            for paper in data.get("data", []):
                authors = paper.get("authors", []) or []
                results.append({
                    "title": paper.get("title", ""),
                    "authors": ", ".join(a.get("name", "") for a in authors[:5]),
                    "year": paper.get("year", 0),
                    "journal": paper.get("venue", ""),
                    "abstract": paper.get("abstract", "") or "",
                    "citation_count": paper.get("citationCount", 0),
                    "source": "semantic_scholar",
                })

            self._cache[cache_key] = results
            return results

        except Exception as e:
            logger.warning("search.semantic_scholar_failed", error=str(e))
            return []

    async def _search_arxiv(
        self,
        query: str,
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """Search arXiv."""
        try:
            cache_key = f"arxiv_{hashlib.md5(query.encode()).hexdigest()}"
            if cache_key in self._cache:
                return self._cache[cache_key]

            await self._rate_limit_delay("arxiv")

            client = await self._get_client()

            url = (
                f"https://export.arxiv.org/api/query"
                f"?search_query=all:{quote(query)}"
                f"&start=0&max_results={min(max_results, 50)}"
                f"&sortBy=relevance&sortOrder=descending"
            )

            resp = await client.get(url)
            # Parse Atom feed (simplified)
            import xml.etree.ElementTree as ET

            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            results = []
            for entry in root.findall("atom:entry", ns):
                title = entry.find("atom:title", ns)
                authors = entry.findall("atom:author/atom:name", ns)
                summary = entry.find("atom:summary", ns)
                published = entry.find("atom:published", ns)

                if title is not None:
                    results.append({
                        "title": title.text[:500] if title.text else "",
                        "authors": ", ".join(a.text[:100] if a.text else "" for a in authors[:5]),
                        "year": published.text[:4] if published is not None and published.text else "",
                        "journal": "arXiv",
                        "abstract": (summary.text[:1000] if summary is not None and summary.text else ""),
                        "source": "arxiv",
                    })

            self._cache[cache_key] = results
            return results

        except Exception as e:
            logger.warning("search.arxiv_failed", error=str(e))
            return []

    async def _search_crossref(
        self,
        query: str,
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """Search CrossRef."""
        try:
            cache_key = f"cr_{hashlib.md5(query.encode()).hexdigest()}"
            if cache_key in self._cache:
                return self._cache[cache_key]

            await self._rate_limit_delay("crossref")

            client = await self._get_client()

            url = (
                f"https://api.crossref.org/works"
                f"?query={quote(query)}"
                f"&rows={min(max_results, 50)}"
                f"&sort=relevance"
            )

            headers = {}
            mailto = self.config.get("crossref_mailto")
            if mailto:
                headers["User-Agent"] = f"CoScientist/1.0 (mailto:{mailto})"

            resp = await client.get(url, headers=headers)
            data = resp.json()

            results = []
            for item in data.get("message", {}).get("items", []):
                authors = item.get("author", []) or []
                author_names = []
                for a in authors[:5]:
                    given = a.get("given", "")
                    family = a.get("family", "")
                    if given and family:
                        author_names.append(f"{given} {family}")
                    elif family:
                        author_names.append(family)

                results.append({
                    "title": item.get("title", [""])[0] if item.get("title") else "",
                    "authors": ", ".join(author_names),
                    "year": item.get("published-print", {}).get("date-parts", [[0]])[0][0]
                        if item.get("published-print") else
                        (item.get("published-online", {}).get("date-parts", [[0]])[0][0] if item.get("published-online") else 0),
                    "journal": item.get("container-title", [""])[0] if item.get("container-title") else "",
                    "doi": item.get("DOI", ""),
                    "citation_count": item.get("is-referenced-by-count", 0),
                    "source": "crossref",
                })

            self._cache[cache_key] = results
            return results

        except Exception as e:
            logger.warning("search.crossref_failed", error=str(e))
            return []

    async def _search_eartharxiv(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search EarthArXiv for climate preprints."""
        try:
            cache_key = f"eartharxiv_{hashlib.md5(query.encode()).hexdigest()}"
            if cache_key in self._cache:
                return self._cache[cache_key]
            
            await self._rate_limit_delay("eartharxiv")
            client = await self._get_client()
            
            # EarthArXiv uses OSF/OAI-PMH or Subject: "Climate"
            url = f"https://api.osf.io/v2/preprints/?filter[subjects]={query}&page[size]={min(max_results, 20)}"
            resp = await client.get(url)
            data = resp.json()
            
            results = []
            for item in data.get("data", []):
                attrs = item.get("attributes", {})
                results.append({
                    "title": attrs.get("title", ""),
                    "authors": "",  # Would parse from relationships
                    "year": attrs.get("date_created", "")[:4] if attrs.get("date_created") else "",
                    "journal": "EarthArXiv",
                    "abstract": attrs.get("description", "")[:1000],
                    "source": "eartharxiv",
                })
            
            self._cache[cache_key] = results
            return results
        except Exception as e:
            logger.warning("search.eartharxiv_failed", error=str(e))
            return []

    async def _rate_limit_delay(self, source: str) -> None:
        """Apply rate limiting per source."""
        now = time.time()
        last = self._last_request_time.get(source, 0)
        delay = max(0, (1.0 / self._rate_limit) - (now - last))
        if delay > 0:
            await asyncio.sleep(delay)
        self._last_request_time[source] = time.time()

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
