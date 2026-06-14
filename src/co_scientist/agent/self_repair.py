"""SelfRepairAgent — diagnose a failed Verify and feed a fix back to Plan
(CLAUDE.md §1.1, was EvolutionAgent)."""

from __future__ import annotations

from typing import List, Optional

import structlog
from pydantic import BaseModel

log = structlog.get_logger("agent.self_repair")


class RepairSuggestion(BaseModel):
    diagnosis: str
    revised_instruction: str


class SelfRepairAgent:
    agent_name = "self_repair"

    def __init__(self, llm_router=None):
        self.llm = llm_router

    async def repair(
        self,
        goal: str,
        failure_reason: str,
        history: Optional[List[str]] = None,
    ) -> RepairSuggestion:
        """Produce a diagnosis + a concrete revised instruction for the next Plan step."""
        if self.llm is not None:
            hist = "\n".join(history or [])[-4000:]
            messages = [
                {
                    "role": "system",
                    "content": "A task attempt failed verification. Diagnose the root "
                    "cause and propose a concrete, different next action. Reply with JSON "
                    '{"diagnosis": str, "revised_instruction": str}.',
                },
                {
                    "role": "user",
                    "content": f"GOAL:\n{goal}\n\nFAILURE:\n{failure_reason}\n\nHISTORY:\n{hist}",
                },
            ]
            try:
                fix = await self.llm.complete(
                    messages, response_format=RepairSuggestion
                )
                if isinstance(fix, RepairSuggestion):
                    return fix
            except Exception as e:  # noqa: BLE001
                log.warning("self_repair.llm_failed", error=str(e))

        return RepairSuggestion(
            diagnosis=f"verification failed: {failure_reason}",
            revised_instruction="Re-examine the last tool output for errors, fix the command or "
            "approach, and try a different concrete step toward the goal.",
        )
