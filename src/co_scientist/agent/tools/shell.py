"""Bash — the primary action verb (CLAUDE.md §4: "prefer it over 10 named wrappers").

All execution flows through the ExecutionSandbox; the Guardian (when present) gates
the command before it runs.
"""

from __future__ import annotations

from typing import Any, Dict

from co_scientist.agent.tools.base import Tool, ToolContext, ToolResult


class BashTool(Tool):
    name = "bash"
    description = (
        "Run a shell command in the sandboxed project workspace and return its "
        "stdout/stderr and exit code. Use this for computation, file manipulation, "
        "running scripts, installing packages, and invoking other CLIs."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": "Optional timeout in seconds.",
            },
        },
        "required": ["command"],
    }

    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        command = args.get("command", "").strip()
        if not command:
            return ToolResult(ok=False, error="bash: empty command")

        # Guardian gates the action before it touches the sandbox (§4, §6).
        if ctx.guardian is not None:
            verdict = await ctx.guardian.check_action(
                action_type="command",
                target=command,
                context={"project_id": ctx.project_id},
            )
            if not verdict.approved:
                return ToolResult(
                    ok=False,
                    error=f"blocked by guardian: {verdict.reason}",
                    metadata={"blocked": True, "risk": verdict.risk},
                )

        if ctx.audit is not None:
            await ctx.audit.record("command", {"command": command}, agent="bash")

        result = await ctx.sandbox.run(
            command, workdir=str(ctx.workdir), timeout=args.get("timeout")
        )
        return ToolResult(
            ok=result.success,
            output=result.output,
            error=result.error,
            metadata={
                "exit_code": result.exit_code,
                "provider": result.provider,
                "execution_time": result.execution_time,
            },
        )
