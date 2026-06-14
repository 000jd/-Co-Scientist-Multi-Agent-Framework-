"""MCP client — connect to Model Context Protocol servers (CLAUDE.md §7 Phase 5).

Uses ``fastmcp`` to expose a remote MCP server's tools to the worker. This is the
documented seam for external tool servers; it requires the ``fastmcp`` package and a
configured server endpoint. Without them the tool degrades to a clear "not configured"
result rather than crashing the loop (CLAUDE.md: "Don't build a custom tool protocol —
use MCP").
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import structlog

from co_scientist.agent.tools.base import Tool, ToolContext, ToolResult

log = structlog.get_logger("agent.tools.mcp")


def fastmcp_available() -> bool:
    try:
        import fastmcp  # noqa: F401

        return True
    except Exception:
        return False


class MCPTool(Tool):
    """Adapter that calls a named tool on a configured MCP server."""

    name = "mcp"
    description = (
        "Call a tool exposed by a connected MCP server. args: {server, tool, arguments}. "
        "Requires fastmcp + a configured server endpoint."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "server": {
                "type": "string",
                "description": "configured MCP server name/url",
            },
            "tool": {"type": "string"},
            "arguments": {"type": "object"},
        },
        "required": ["server", "tool"],
    }

    def __init__(self, servers: Optional[Dict[str, str]] = None):
        self.servers = servers or {}

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        if not fastmcp_available():
            return ToolResult(
                ok=False,
                error="mcp: fastmcp not installed (pip install fastmcp)",
                metadata={"configured": False},
            )
        server = self.servers.get(args.get("server", ""), args.get("server", ""))
        if not server:
            return ToolResult(
                ok=False,
                error="mcp: no server endpoint configured",
                metadata={"configured": False},
            )
        try:
            from fastmcp import Client  # type: ignore

            async def _call():
                async with Client(server) as client:
                    return await client.call_tool(
                        args["tool"], args.get("arguments", {})
                    )

            result = await asyncio.wait_for(_call(), timeout=120)
            return ToolResult(ok=True, output=str(result))
        except Exception as e:  # noqa: BLE001
            log.warning("mcp.call_failed", error=str(e))
            return ToolResult(ok=False, error=f"mcp call failed: {e}")

    async def list_server_tools(self, server: str) -> List[str]:
        if not fastmcp_available():
            return []
        try:
            from fastmcp import Client  # type: ignore

            async with Client(self.servers.get(server, server)) as client:
                tools = await client.list_tools()
                return [t.name for t in tools]
        except Exception:
            return []
