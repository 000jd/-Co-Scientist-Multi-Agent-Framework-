"""VerifierAgent — the explicit Verify stage (CLAUDE.md §1.1, was ReflectionAgent).

"No implicit 'output exists = success'." Verification is, in order of preference:

1. a deterministic shell check (``verification`` command; exit 0 == pass), or
2. an LLM judge that reads the goal + answer + observations, or
3. a conservative heuristic when neither is available.
"""

from __future__ import annotations

from typing import List, Optional

import structlog
from pydantic import BaseModel

log = structlog.get_logger("agent.verifier")


class VerificationResult(BaseModel):
    passed: bool
    reason: str
    method: str  # "command" | "llm" | "heuristic"


class _LLMVerdict(BaseModel):
    passed: bool
    reason: str


class VerifierAgent:
    agent_name = "verifier"

    def __init__(self, llm_router=None, sandbox=None, guardian=None, audit=None):
        self.llm = llm_router
        self.sandbox = sandbox
        self.guardian = guardian
        self.audit = audit

    async def verify(
        self,
        goal: str,
        answer: str,
        observations: Optional[List[str]] = None,
        verification_cmd: Optional[str] = None,
        workdir: Optional[str] = None,
    ) -> VerificationResult:
        # 1. Deterministic shell check (preferred — no LLM, fully reproducible).
        # The command is model-supplied, so it is gated by the Guardian like any
        # other outbound action (CLAUDE.md §4 — never bypass the Guardian).
        if verification_cmd and self.sandbox is not None:
            if self.guardian is not None:
                verdict = await self.guardian.check_action(
                    "command", verification_cmd, {"stage": "verify"}
                )
                if not verdict.approved:
                    return VerificationResult(
                        passed=False,
                        reason=f"verification command blocked by guardian: {verdict.reason}",
                        method="command",
                    )
            if self.audit is not None:
                await self.audit.record(
                    "command", {"command": verification_cmd}, agent="verifier"
                )
            result = await self.sandbox.run(verification_cmd, workdir=workdir)
            passed = result.success
            return VerificationResult(
                passed=passed,
                reason=f"verification command exit={result.exit_code}: {result.output[:300]}",
                method="command",
            )

        # 2. LLM judge.
        if self.llm is not None:
            obs = "\n".join(observations or [])[-4000:]
            messages = [
                {
                    "role": "system",
                    "content": "You are a strict verifier. Decide whether the ANSWER "
                    "actually satisfies the GOAL given the OBSERVATIONS. Reply with JSON "
                    '{"passed": bool, "reason": str}. Be skeptical: a plausible-looking answer that '
                    "was not actually computed/produced must fail.",
                },
                {
                    "role": "user",
                    "content": f"GOAL:\n{goal}\n\nANSWER:\n{answer}\n\nOBSERVATIONS:\n{obs}",
                },
            ]
            try:
                verdict = await self.llm.complete(messages, response_format=_LLMVerdict)
                if isinstance(verdict, _LLMVerdict):
                    return VerificationResult(
                        passed=verdict.passed, reason=verdict.reason, method="llm"
                    )
            except Exception as e:  # noqa: BLE001
                log.warning("verifier.llm_failed", error=str(e))

        # 3. Conservative heuristic.
        passed = bool(answer and answer.strip())
        return VerificationResult(
            passed=passed,
            reason="non-empty answer (heuristic; no verifier available)"
            if passed
            else "empty answer",
            method="heuristic",
        )
