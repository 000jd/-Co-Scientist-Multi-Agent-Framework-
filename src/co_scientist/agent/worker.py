"""The 5-stage worker loop — the unit of all useful work (CLAUDE.md §1.1).

    Think → Plan → Act → Observe → Verify
                                      │
                            pass ─────┘──► mark done, emit artifact
                            fail ─────────► SelfRepairAgent → loop back to Plan

Loop terminates ONLY on FINAL_ANSWER (verified), max_turns, budget exceeded, or a
stop signal — never on "the model produced some text" (the v0.1 regression).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import structlog
from pydantic import BaseModel

from co_scientist.agent.context_compressor import ContextCompressor
from co_scientist.agent.protocol import parse_action
from co_scientist.agent.self_repair import SelfRepairAgent
from co_scientist.agent.tools.base import Tool, ToolContext
from co_scientist.agent.verifier import VerifierAgent

log = structlog.get_logger("agent.worker")

_DEFAULT_SYSTEM = """You are JARVIS, an autonomous agent that completes real tasks by acting, not by \
describing actions. You operate a sandboxed Linux workspace.

On every turn, reason briefly, then emit EXACTLY ONE of:

  tool_call({"tool": "<name>", "args": { ... }})
  FINAL_ANSWER({"answer": "<the answer or deliverable summary>", "verification": "<optional shell \
command that exits 0 iff the task is truly done>"})

Rules:
- Bash is your primary tool. Prefer it over everything else.
- Do the actual work (compute, write files, run code). Never claim a result you did not produce.
- When the task is genuinely complete, emit FINAL_ANSWER. Include a `verification` shell command \
whenever the result can be checked deterministically.
- One action per turn. Wait for the OBSERVATION before the next step.
"""


class WorkerResult(BaseModel):
    success: bool
    answer: str = ""
    turns: int = 0
    verified: bool = False
    verification_reason: str = ""
    verification_method: str = ""
    artifacts: List[str] = []
    transcript: List[Dict[str, str]] = []
    stop_reason: str = (
        "unknown"  # final_answer | max_turns | budget | stopped | no_progress | error
    )


class Worker:
    """A single 5-stage ReAct worker. One worker completes one node/goal."""

    def __init__(
        self,
        llm_router,
        sandbox,
        tools: Optional[Dict[str, Tool]] = None,
        *,
        verifier: Optional[VerifierAgent] = None,
        self_repair: Optional[SelfRepairAgent] = None,
        compressor: Optional[ContextCompressor] = None,
        guardian=None,
        audit=None,
        max_turns: int = 50,
        budget_usd: float = 5.0,
        project_id: str = "default",
        workdir: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.llm = llm_router
        self.sandbox = sandbox
        if tools is None:
            from co_scientist.agent.tools.file_ops import default_tools

            tools = default_tools()
        self.tools = tools
        self.verifier = verifier or VerifierAgent(
            llm_router=llm_router, sandbox=sandbox, guardian=guardian, audit=audit
        )
        self.self_repair = self_repair or SelfRepairAgent(llm_router=llm_router)
        self.compressor = compressor or ContextCompressor(llm_router=llm_router)
        self.guardian = guardian
        self.audit = audit
        self.max_turns = max_turns
        self.budget_usd = budget_usd
        self.project_id = project_id
        self.workdir = Path(workdir) if workdir else Path("./projects") / project_id
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.system_prompt = system_prompt or _load_system_prompt()
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    def _spent_usd(self) -> float:
        tracker = getattr(self.llm, "cost_tracker", None)
        return float(getattr(tracker, "current_total", 0.0)) if tracker else 0.0

    def _tool_catalogue(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in self.tools.values())

    async def run(
        self, goal: str, context: str = "", verification: Optional[str] = None
    ) -> WorkerResult:
        # Per-worker budget: measure THIS worker's spend as a delta from start, so
        # parallel workers sharing one cost tracker don't starve each other (review #0).
        self._start_spent = self._spent_usd()
        ctx = ToolContext(
            workdir=self.workdir,
            sandbox=self.sandbox,
            project_id=self.project_id,
            guardian=self.guardian,
            audit=self.audit,
        )
        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": f"{self.system_prompt}\n\nAVAILABLE TOOLS:\n{self._tool_catalogue()}",
            },
            {
                "role": "user",
                "content": f"TASK:\n{goal}"
                + (f"\n\nCONTEXT:\n{context}" if context else ""),
            },
        ]
        observations: List[str] = []
        artifacts: List[str] = []
        last_answer = ""

        for turn in range(1, self.max_turns + 1):
            if self._stop:
                return self._result(
                    False, last_answer, turn, "stopped", messages, artifacts
                )
            if self._spent_usd() - self._start_spent > self.budget_usd:
                return self._result(
                    False, last_answer, turn, "budget", messages, artifacts
                )

            messages = await self.compressor.maybe_compress(messages)

            # Think + Plan
            try:
                plan_text = await self.llm.complete(messages)
            except Exception as e:  # noqa: BLE001
                log.warning("worker.llm_failed", error=str(e), turn=turn)
                return self._result(
                    False, last_answer, turn, "error", messages, artifacts
                )
            if not isinstance(plan_text, str):
                plan_text = str(plan_text)
            messages.append({"role": "assistant", "content": plan_text})
            action = parse_action(plan_text)

            if action.kind == "final_answer":
                last_answer = action.answer or ""
                # A planner-supplied node verification (from the DAG) is authoritative
                # and takes precedence over the worker's own (review #1).
                verification_cmd = verification or action.verification
                verdict = await self.verifier.verify(
                    goal=goal,
                    answer=last_answer,
                    observations=observations,
                    verification_cmd=verification_cmd,
                    workdir=str(self.workdir),
                )
                if verdict.passed:
                    return WorkerResult(
                        success=True,
                        answer=last_answer,
                        turns=turn,
                        verified=True,
                        verification_reason=verdict.reason,
                        verification_method=verdict.method,
                        artifacts=artifacts,
                        transcript=messages,
                        stop_reason="final_answer",
                    )
                # Verify failed → SelfRepair → loop back to Plan
                fix = await self.self_repair.repair(goal, verdict.reason, observations)
                messages.append(
                    {
                        "role": "user",
                        "content": f"[VERIFY FAILED] {verdict.reason}\n[DIAGNOSIS] {fix.diagnosis}\n"
                        f"[NEXT] {fix.revised_instruction}\nDo not emit FINAL_ANSWER again "
                        f"until the work is actually done.",
                    }
                )
                continue

            if action.kind == "tool_call":
                tool = self.tools.get(action.tool or "")
                if tool is None:
                    messages.append(
                        {
                            "role": "user",
                            "content": f"[OBSERVATION] unknown tool '{action.tool}'. Available: "
                            f"{', '.join(self.tools)}",
                        }
                    )
                    continue
                # Act
                result = await tool.run(action.args, ctx)
                # Observe
                obs = (
                    result.output
                    if result.ok
                    else f"ERROR: {result.error}\n{result.output}"
                )
                observations.append(obs)
                artifacts.extend(result.artifacts)
                messages.append(
                    {"role": "user", "content": f"[OBSERVATION] {obs[:6000]}"}
                )
                continue

            # noop — nudge once, fail on repeated no-progress
            messages.append(
                {
                    "role": "user",
                    "content": "[SYSTEM] No actionable tool_call or FINAL_ANSWER found. Emit exactly one.",
                }
            )
            if _trailing_noops(messages) >= 3:
                return self._result(
                    False, last_answer, turn, "no_progress", messages, artifacts
                )

        return self._result(
            False, last_answer, self.max_turns, "max_turns", messages, artifacts
        )

    def _result(
        self, success, answer, turns, stop_reason, messages, artifacts
    ) -> WorkerResult:
        return WorkerResult(
            success=success,
            answer=answer,
            turns=turns,
            verified=False,
            stop_reason=stop_reason,
            transcript=messages,
            artifacts=artifacts,
        )


def _trailing_noops(messages: List[Dict[str, str]]) -> int:
    n = 0
    for m in reversed(messages):
        if m.get("role") == "user" and m.get("content", "").startswith(
            "[SYSTEM] No actionable"
        ):
            n += 1
        elif m.get("role") == "assistant":
            continue
        else:
            break
    return n


def _load_system_prompt() -> str:
    prompt_path = Path(__file__).parent / "prompts" / "system.md"
    try:
        return prompt_path.read_text(encoding="utf-8")
    except Exception:
        return _DEFAULT_SYSTEM
