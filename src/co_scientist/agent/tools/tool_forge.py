"""DynamicToolForge — find-or-forge tools at runtime (CLAUDE.md §1.3, §1.4, §7 Phase 4).

When the worker needs a capability no built-in tool provides:

  1. vector/keyword-search the ToolLibrary for an existing tool and reuse it; else
  2. ask the LLM to write a self-contained script tool;
  3. write the script into the sandbox workspace and run its sample input — on a
     mismatch/error, loop a self-repair fix prompt up to ``max_repair_attempts`` times;
  4. on success persist the spec to the library so it is forged once and reused next time.

All script execution flows through the ExecutionSandbox; nothing runs on the host.
"""

from __future__ import annotations

import re
import shlex
from typing import Any, Dict, Optional

import structlog
from pydantic import BaseModel, Field

from co_scientist.agent.tools.base import ToolContext
from co_scientist.agent.tools.tool_library import ToolLibrary, ToolSpec, _keywords
from co_scientist.infrastructure.execution_sandbox import ExecResult

log = structlog.get_logger("agent.tools.tool_forge")

# A library hit is "good enough" to reuse when its description shares at least this
# fraction of the query's content words (deterministic; no embeddings required).
_REUSE_OVERLAP_RATIO = 0.5

# Run-command templates keyed by validated language. The model never supplies the
# entrypoint: it is derived here so an LLM-authored string can't inject shell (#8).
_RUN_TEMPLATES = {
    "python": "python3 {script_path} {args}",
    "bash": "bash {script_path} {args}",
    "sh": "bash {script_path} {args}",
    "shell": "bash {script_path} {args}",
    "node": "node {script_path} {args}",
    "javascript": "node {script_path} {args}",
}
_DEFAULT_LANGUAGE = "python"


def _normalize_language(language: str) -> str:
    """Map a (possibly model-supplied) language to a supported one; default python."""
    lang = (language or "").strip().lower()
    return lang if lang in _RUN_TEMPLATES else _DEFAULT_LANGUAGE


def _run_command(language: str, script_path: str, args: str) -> str:
    """Build the run command from a FIXED per-language template (no model input)."""
    template = _RUN_TEMPLATES[_normalize_language(language)]
    return template.format(script_path=shlex.quote(script_path), args=args)


class ForgedTool(BaseModel):
    """Structured LLM output describing a new script tool."""

    name: str
    description: str
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    language: str = "python"
    script: str
    sample_input: str = "{}"  # JSON string of the args used to exercise the tool
    expected_output: str = ""


class _RepairedTool(BaseModel):
    script: str


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", name.strip().lower()).strip("_")
    return s or "forged_tool"


def _ext(language: str) -> str:
    return {
        "python": "py",
        "bash": "sh",
        "shell": "sh",
        "node": "js",
        "javascript": "js",
    }.get(language.lower(), "txt")


def _args_string(args: Dict[str, Any]) -> str:
    """Render a flat args dict as positional shell arguments (stable key order)."""
    parts = []
    for key in sorted(args):
        parts.append(shlex.quote(str(args[key])))
    return " ".join(parts)


def _overlap_ratio(query: str, spec: ToolSpec) -> float:
    q = set(_keywords(query))
    if not q:
        return 0.0
    doc = set(_keywords(f"{spec.name} {spec.description}"))
    return len(q & doc) / len(q)


class DynamicToolForge:
    """Finds an existing tool for a task or forges, tests and registers a new one."""

    def __init__(
        self,
        llm_router,
        sandbox,
        library: ToolLibrary,
        max_repair_attempts: int = 5,
    ):
        self.llm = llm_router
        self.sandbox = sandbox
        self.library = library
        self.max_repair_attempts = max_repair_attempts

    # ── public API ──────────────────────────────────────────────────────────
    async def find_or_forge(
        self, task_description: str, ctx: ToolContext
    ) -> Optional[ToolSpec]:
        """Return a tool for ``task_description``, reusing the library when possible."""
        hits = self.library.search(task_description, top_k=3)
        for spec in hits:
            if _overlap_ratio(task_description, spec) >= _REUSE_OVERLAP_RATIO:
                log.info("tool_forge.reuse", name=spec.name, task=task_description[:80])
                return spec

        if self.llm is None:
            log.warning("tool_forge.no_llm", task=task_description[:80])
            return None

        spec = await self._forge(task_description, ctx)
        if spec is not None:
            self.library.save(spec)
            log.info("tool_forge.registered", name=spec.name)
        return spec

    async def run_tool(
        self, spec: ToolSpec, args: Dict[str, Any], ctx: ToolContext
    ) -> ExecResult:
        """Materialise ``spec``'s script in the workspace and run it with ``args``."""
        script_path = await self._write_script(
            spec.name, spec.language, spec.script, ctx
        )
        command = _run_command(spec.language, script_path, _args_string(args))
        return await self._gated_run(command, ctx)

    # ── forging ─────────────────────────────────────────────────────────────
    async def _forge(
        self, task_description: str, ctx: ToolContext
    ) -> Optional[ToolSpec]:
        try:
            forged = await self.llm.complete(
                self._forge_messages(task_description), response_format=ForgedTool
            )
        except Exception as e:  # noqa: BLE001
            log.warning("tool_forge.llm_failed", error=str(e))
            return None
        if not isinstance(forged, ForgedTool):
            return None

        name = _slug(forged.name)
        sample_input = _parse_json(forged.sample_input)
        language = _normalize_language(forged.language)
        spec = ToolSpec(
            name=name,
            description=forged.description or task_description,
            input_schema=forged.input_schema,
            script=forged.script,
            language=language,
            entrypoint=_RUN_TEMPLATES[language],
            sample_input=sample_input,
            sample_output=forged.expected_output,
        )

        result = await self._test(spec, sample_input, ctx)
        if self._passes(result, forged.expected_output):
            spec.sample_output = result.output.strip() or spec.sample_output
            return spec

        # Self-repair loop: feed the failure back to the LLM for up to N fixes.
        for attempt in range(1, self.max_repair_attempts + 1):
            log.info("tool_forge.repair_attempt", name=name, attempt=attempt)
            repaired = await self._repair(
                spec, sample_input, result, forged.expected_output
            )
            if repaired is None:
                break
            spec.script = repaired.script
            result = await self._test(spec, sample_input, ctx)
            if self._passes(result, forged.expected_output):
                spec.sample_output = result.output.strip() or spec.sample_output
                return spec

        log.warning(
            "tool_forge.forge_failed",
            name=name,
            error=result.error or "output mismatch",
        )
        return None

    async def _test(
        self, spec: ToolSpec, sample_input: Dict[str, Any], ctx: ToolContext
    ) -> ExecResult:
        script_path = await self._write_script(
            spec.name, spec.language, spec.script, ctx
        )
        command = _run_command(spec.language, script_path, _args_string(sample_input))
        return await self._gated_run(command, ctx)

    async def _gated_run(self, command: str, ctx: ToolContext) -> ExecResult:
        """Run ``command`` only after a Guardian approval, recording it to the audit log."""
        if ctx.guardian is not None:
            decision = await ctx.guardian.check_action(
                "command", command, {"project_id": ctx.project_id}
            )
            if not getattr(decision, "approved", False):
                if ctx.audit is not None:
                    await ctx.audit.record(
                        "tool_forge_blocked",
                        {"command": command, "project_id": ctx.project_id},
                    )
                return ExecResult(
                    success=False,
                    error=f"guardian blocked command: {command[:120]!r}",
                    provider=getattr(self.sandbox, "provider", "local"),
                )
        if ctx.audit is not None:
            await ctx.audit.record(
                "command", {"command": command, "project_id": ctx.project_id}
            )
        return await self.sandbox.run(command, workdir=str(ctx.workdir))

    async def _repair(
        self,
        spec: ToolSpec,
        sample_input: Dict[str, Any],
        result: ExecResult,
        expected_output: str,
    ) -> Optional[_RepairedTool]:
        messages = [
            {
                "role": "system",
                "content": "You wrote a script tool that failed its test. Fix the script so it "
                "produces the expected output for the sample input. Reply with JSON "
                '{"script": str}. Keep the script self-contained.',
            },
            {
                "role": "user",
                "content": (
                    f"TOOL: {spec.name}\nLANGUAGE: {spec.language}\n"
                    f"SAMPLE_INPUT: {sample_input}\n"
                    f"EXPECTED_OUTPUT: {expected_output!r}\n"
                    f"ACTUAL_OUTPUT: {result.output!r}\nERROR: {result.error}\n\n"
                    f"CURRENT SCRIPT:\n{spec.script}"
                ),
            },
        ]
        try:
            fix = await self.llm.complete(messages, response_format=_RepairedTool)
        except Exception as e:  # noqa: BLE001
            log.warning("tool_forge.repair_llm_failed", error=str(e))
            return None
        return fix if isinstance(fix, _RepairedTool) else None

    # ── helpers ─────────────────────────────────────────────────────────────
    async def _write_script(
        self, name: str, language: str, script: str, ctx: ToolContext
    ) -> str:
        rel = f".tools/{name}.{_ext(language)}"
        target = ctx.resolve(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(script, encoding="utf-8")
        return str(target)

    def _passes(self, result: ExecResult, expected_output: str) -> bool:
        if not result.success:
            return False
        expected = (expected_output or "").strip()
        if not expected:
            return True
        return expected in result.output.strip()

    def _forge_messages(self, task_description: str):
        return [
            {
                "role": "system",
                "content": (
                    "Write a SINGLE self-contained command-line script that performs the requested "
                    "task. It reads its inputs from positional command-line arguments (in sorted key "
                    "order) and prints ONLY the result to stdout. The runtime is chosen from the "
                    "'language' field (python, bash, or node); you do NOT specify how it is run. "
                    "Reply with JSON matching: "
                    '{"name": str, "description": str, "input_schema": object, '
                    '"language": str (one of: python, bash, node), "script": str, '
                    '"sample_input": str (JSON object of args), "expected_output": str}.'
                ),
            },
            {"role": "user", "content": f"TASK:\n{task_description}"},
        ]


def _parse_json(text: str) -> Dict[str, Any]:
    import json

    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}
