"""Web tools — search + fetch. External content is ALWAYS wrapped in the
PromptInjectionGuard before it re-enters the model (CLAUDE.md §10)."""

from __future__ import annotations

import asyncio
from typing import Any, Dict

from co_scientist.agent.tools.base import Tool, ToolContext, ToolResult


class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web (DuckDuckGo, no API key) and return ranked title/url/snippet results."
    input_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
        "required": ["query"],
    }

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        query = args.get("query", "").strip()
        if not query:
            return ToolResult(ok=False, error="web_search: empty query")
        n = int(args.get("max_results", 5))
        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(_ddg_search, query, n), timeout=30
            )
        except Exception as e:  # noqa: BLE001
            return ToolResult(ok=False, error=f"web_search failed: {e}")
        if not results:
            return ToolResult(ok=True, output="(no results)")
        lines = [
            f"{i + 1}. {r['title']}\n   {r['url']}\n   {r['snippet']}"
            for i, r in enumerate(results)
        ]
        guarded = await _guard(ctx, "\n".join(lines), source="web_search")
        return ToolResult(ok=True, output=guarded, metadata={"count": len(results)})


class WebFetchTool(Tool):
    name = "web_fetch"
    description = "Fetch a URL and return its readable text content (guarded against prompt injection)."
    input_schema = {
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    }

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        url = args.get("url", "").strip()
        if not url.startswith(("http://", "https://")):
            return ToolResult(
                ok=False, error="web_fetch: url must start with http(s)://"
            )
        from co_scientist.agent.tools._netguard import is_blocked_url

        blocked = is_blocked_url(url)
        if blocked:
            return ToolResult(ok=False, error=f"web_fetch: blocked url ({blocked})")
        try:
            from co_scientist.search.web_reader import WebReader

            text = await asyncio.wait_for(
                WebReader(max_length=8000).fetch(url), timeout=30
            )
        except Exception as e:  # noqa: BLE001
            return ToolResult(ok=False, error=f"web_fetch failed: {e}")
        if not text:
            return ToolResult(ok=False, error="web_fetch: no content")
        guarded = await _guard(ctx, text, source=url)
        return ToolResult(ok=True, output=guarded)


def _ddg_search(query: str, n: int):
    from duckduckgo_search import DDGS

    out = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=n):
            out.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
            )
    return out


async def _guard(ctx: ToolContext, content: str, source: str) -> str:
    """Wrap external content so the model treats it as untrusted data (§10)."""
    try:
        from co_scientist.governance.prompt_injection_guard import PromptInjectionGuard

        return await PromptInjectionGuard(strict=True).sanitize(
            content, source_url=source
        )
    except Exception:
        return f"<external_content source={source!r}>\n{content}\n</external_content>"


def web_tools() -> Dict[str, Tool]:
    tools = [WebSearchTool(), WebFetchTool()]
    return {t.name: t for t in tools}
