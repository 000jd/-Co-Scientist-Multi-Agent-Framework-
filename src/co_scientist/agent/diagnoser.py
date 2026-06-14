"""DiagnoseAgent — post-mortem "what went wrong" (CLAUDE.md §1.4/§3, was MetaReviewAgent).

Runs when a task terminates without success (max_turns / no_progress / budget) to produce
a human-readable root-cause summary. LLM-backed with a deterministic heuristic fallback.
"""

from __future__ import annotations

from typing import List, Optional

import structlog
from pydantic import BaseModel

log = structlog.get_logger("agent.diagnoser")


class Diagnosis(BaseModel):
    root_cause: str
    suggestion: str


class DiagnoseAgent:
    agent_name = "diagnoser"

    def __init__(self, llm_router=None):
        self.llm = llm_router

    async def diagnose(
        self, goal: str, stop_reason: str, transcript: Optional[List[dict]] = None
    ) -> Diagnosis:
        tail = "\n".join(m.get("content", "")[:300] for m in (transcript or [])[-8:])
        if self.llm is not None:
            try:
                d = await self.llm.complete(
                    [
                        {
                            "role": "system",
                            "content": "A task stopped without success. Give the root cause "
                            'and one concrete suggestion. Reply JSON {"root_cause": str, "suggestion": str}.',
                        },
                        {
                            "role": "user",
                            "content": f"GOAL: {goal}\nSTOP: {stop_reason}\nTAIL:\n{tail}",
                        },
                    ],
                    response_format=Diagnosis,
                )
                if isinstance(d, Diagnosis):
                    return d
            except Exception as e:  # noqa: BLE001
                log.warning("diagnoser.llm_failed", error=str(e))

        reasons = {
            "max_turns": (
                "hit the turn cap before finishing",
                "raise agent.max_turns or narrow the task",
            ),
            "budget": (
                "exceeded the dollar budget",
                "raise agent.budget_usd or use a cheaper model",
            ),
            "no_progress": (
                "kept emitting unactionable output",
                "make the goal more concrete",
            ),
            "error": (
                "an LLM/tool error interrupted the loop",
                "check provider config and retry",
            ),
            "stopped": ("was stopped externally", "resume from the audit log"),
        }
        rc, sug = reasons.get(
            stop_reason, (f"stopped: {stop_reason}", "inspect the transcript")
        )
        return Diagnosis(root_cause=rc, suggestion=sug)
