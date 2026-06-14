"""File tools (Read / Write / Edit / Glob / Grep), all scoped to the project
workspace (CLAUDE.md §4: never access the host filesystem outside ``projects/<id>/``).
"""

from __future__ import annotations

import re
from typing import Any, Dict

from co_scientist.agent.tools.base import Tool, ToolContext, ToolResult


class ReadTool(Tool):
    name = "read_file"
    description = "Read a UTF-8 text file from the project workspace."
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            target = ctx.resolve(args["path"])
            text = target.read_text(encoding="utf-8")
            if ctx.audit is not None:
                await ctx.audit.record(
                    "file_read", {"path": str(target)}, agent="read_file"
                )
            return ToolResult(ok=True, output=text)
        except Exception as e:  # noqa: BLE001
            return ToolResult(ok=False, error=str(e))


class WriteTool(Tool):
    name = "write_file"
    description = "Write (create/overwrite) a text file in the project workspace."
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    }

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            target = ctx.resolve(args["path"])
            if ctx.guardian is not None:
                verdict = await ctx.guardian.check_action(
                    action_type="file_write",
                    target=str(target),
                    context={"project_id": ctx.project_id},
                )
                if not verdict.approved:
                    return ToolResult(
                        ok=False, error=f"blocked by guardian: {verdict.reason}"
                    )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(args["content"], encoding="utf-8")
            if ctx.audit is not None:
                await ctx.audit.record(
                    "file_write",
                    {"path": str(target), "bytes": len(args["content"])},
                    agent="write_file",
                )
            return ToolResult(
                ok=True,
                output=f"wrote {len(args['content'])} bytes to {args['path']}",
                artifacts=[str(target)],
            )
        except Exception as e:  # noqa: BLE001
            return ToolResult(ok=False, error=str(e))


class EditTool(Tool):
    name = "edit_file"
    description = (
        "Replace the first occurrence of old_text with new_text in a workspace file."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_text": {"type": "string"},
            "new_text": {"type": "string"},
        },
        "required": ["path", "old_text", "new_text"],
    }

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            target = ctx.resolve(args["path"])
            if (
                ctx.guardian is not None
            ):  # gate the mutation like WriteTool (review #17)
                verdict = await ctx.guardian.check_action(
                    action_type="edit_file",
                    target=str(target),
                    context={"project_id": ctx.project_id},
                )
                if not verdict.approved:
                    return ToolResult(
                        ok=False, error=f"blocked by guardian: {verdict.reason}"
                    )
            text = target.read_text(encoding="utf-8")
            if args["old_text"] not in text:
                return ToolResult(ok=False, error="old_text not found in file")
            target.write_text(
                text.replace(args["old_text"], args["new_text"], 1), encoding="utf-8"
            )
            if ctx.audit is not None:
                await ctx.audit.record(
                    "file_write", {"path": str(target), "edit": True}, agent="edit_file"
                )
            return ToolResult(
                ok=True, output=f"edited {args['path']}", artifacts=[str(target)]
            )
        except Exception as e:  # noqa: BLE001
            return ToolResult(ok=False, error=str(e))


class GlobTool(Tool):
    name = "glob"
    description = "List workspace files matching a glob pattern (e.g. **/*.py)."
    input_schema = {
        "type": "object",
        "properties": {"pattern": {"type": "string"}},
        "required": ["pattern"],
    }

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            matches = sorted(
                str(p.relative_to(ctx.workdir))
                for p in ctx.workdir.glob(args["pattern"])
                if p.is_file()
            )
            return ToolResult(ok=True, output="\n".join(matches) or "(no matches)")
        except Exception as e:  # noqa: BLE001
            return ToolResult(ok=False, error=str(e))


class GrepTool(Tool):
    name = "grep"
    description = (
        "Search workspace files for a regex; returns matching file:line entries."
    )
    input_schema = {
        "type": "object",
        "properties": {"pattern": {"type": "string"}, "glob": {"type": "string"}},
        "required": ["pattern"],
    }

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            rx = re.compile(args["pattern"])
            glob = args.get("glob", "**/*")
            hits = []
            for p in ctx.workdir.glob(glob):
                if not p.is_file():
                    continue
                try:
                    for i, line in enumerate(
                        p.read_text(encoding="utf-8").splitlines(), 1
                    ):
                        if rx.search(line):
                            hits.append(
                                f"{p.relative_to(ctx.workdir)}:{i}: {line.strip()[:200]}"
                            )
                except (UnicodeDecodeError, OSError):
                    continue
            return ToolResult(ok=True, output="\n".join(hits[:200]) or "(no matches)")
        except Exception as e:  # noqa: BLE001
            return ToolResult(ok=False, error=str(e))


def default_tools(include_web: bool = True) -> Dict[str, Tool]:
    """The built-in tool set every worker starts with: bash + file ops (+ web)."""
    from co_scientist.agent.tools.shell import BashTool

    tools: Dict[str, Tool] = {
        t.name: t
        for t in (
            BashTool(),
            ReadTool(),
            WriteTool(),
            EditTool(),
            GlobTool(),
            GrepTool(),
        )
    }
    if include_web:
        try:
            from co_scientist.agent.tools.web import web_tools

            tools.update(web_tools())
        except Exception:
            pass  # web tools are optional (need duckduckgo-search)
    return tools
