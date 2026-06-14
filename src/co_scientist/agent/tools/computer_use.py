"""Computer-use tool — browser navigate / extract / screenshot fallback
(CLAUDE.md §3, §7 Phase 5). Wraps the existing Playwright LocalActionBridge, which
itself falls back to httpx when a browser isn't available."""

from __future__ import annotations

from typing import Any, Dict

import structlog

from co_scientist.agent.tools.base import Tool, ToolContext, ToolResult

log = structlog.get_logger("agent.tools.computer_use")


class ComputerUseTool(Tool):
    name = "computer_use"
    description = (
        "Drive a headless browser: navigate to a URL, optionally click a CSS selector and "
        "extract the resulting page text. args: {url, selector?, action?}."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "selector": {"type": "string"},
            "action": {"type": "string", "enum": ["navigate", "click_extract"]},
        },
        "required": ["url"],
    }

    def __init__(self, bridge=None):
        self._bridge = bridge

    async def _get_bridge(self):
        if self._bridge is None:
            from co_scientist.infrastructure.local_action_bridge import (
                LocalActionBridge,
            )

            self._bridge = LocalActionBridge()
            await self._bridge.start()
        return self._bridge

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        url = args.get("url", "").strip()
        if not url.startswith(("http://", "https://")):
            return ToolResult(
                ok=False, error="computer_use: url must start with http(s)://"
            )
        from co_scientist.agent.tools._netguard import is_blocked_url

        blocked = is_blocked_url(url)
        if blocked:
            return ToolResult(ok=False, error=f"computer_use: blocked url ({blocked})")
        try:
            bridge = await self._get_bridge()
            if args.get("action") == "click_extract" and args.get("selector"):
                res = await bridge.click_and_extract(url, args["selector"])
            else:
                res = await bridge.navigate(url)
        except Exception as e:  # noqa: BLE001
            return ToolResult(ok=False, error=f"computer_use failed: {e}")

        if not getattr(res, "success", False):
            return ToolResult(
                ok=False, error=getattr(res, "error", "navigation failed")
            )

        content = await _guard(getattr(res, "content", ""), url)
        return ToolResult(ok=True, output=f"[{getattr(res, 'title', '')}]\n{content}")


async def _guard(content: str, source: str) -> str:
    try:
        from co_scientist.governance.prompt_injection_guard import PromptInjectionGuard

        return await PromptInjectionGuard(strict=True).sanitize(
            content, source_url=source
        )
    except Exception:
        return content
