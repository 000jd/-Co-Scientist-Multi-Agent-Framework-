"""GuardianAgent — wraps EVERY outbound action (CLAUDE.md §4, §6).

BashTool / WriteTool / task_runner call ``check_action`` before any side effect.
The Guardian classifies risk, audits the decision, and approves or blocks.

Contract (must not change):
    guardian = GuardianAgent(audit_logger=audit)
    verdict = await guardian.check_action(action_type, target, context) -> GateDecision
"""

from __future__ import annotations

import asyncio
import structlog
from typing import Any, Dict, Optional

from pydantic import BaseModel

from co_scientist.governance.permission import ActionRisk, Permission

logger = structlog.get_logger("guardian")


class GateDecision(BaseModel):
    approved: bool
    risk: str
    reason: str
    modified_target: Optional[str] = None


class GuardianAgent:
    """Gates outbound actions. READ/LOW pass; MEDIUM passes but is audited;
    HIGH is blocked unless explicitly allowed or a human approves (interactive)."""

    def __init__(
        self,
        audit_logger: Any = None,
        permission: Optional[Permission] = None,
        interactive: bool = False,
        auto_approve_threshold: float = 0.65,
    ):
        self.audit_logger = audit_logger
        self.permission = permission or Permission()
        self.interactive = interactive
        self.auto_approve_threshold = auto_approve_threshold

    async def check_action(
        self,
        action_type: str,
        target: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> GateDecision:
        context = context or {}
        risk = self.permission.classify(action_type, target)

        if risk in (ActionRisk.READ, ActionRisk.LOW):
            decision = GateDecision(
                approved=True, risk=risk.value, reason=f"{risk.value} action permitted"
            )

        elif risk == ActionRisk.MEDIUM:
            decision = GateDecision(
                approved=True, risk=risk.value, reason="medium-risk action audited"
            )

        else:  # HIGH
            decision = await self._gate_high(action_type, target)

        await self._audit(action_type, target, decision)
        return decision

    async def _gate_high(self, action_type: str, target: str) -> GateDecision:
        # An explicit allow overrides the block.
        if self.permission.is_allowed(action_type, target):
            return GateDecision(
                approved=True,
                risk=ActionRisk.HIGH.value,
                reason="high-risk action explicitly allow-listed",
            )

        # Interactive: ask the human (off the event loop).
        if self.interactive:
            approved = await self._prompt_human(action_type, target)
            if approved:
                return GateDecision(
                    approved=True,
                    risk=ActionRisk.HIGH.value,
                    reason="high-risk action approved by human",
                )

        return GateDecision(
            approved=False,
            risk=ActionRisk.HIGH.value,
            reason=f"high-risk {action_type} blocked: {target[:120]!r}",
        )

    async def _prompt_human(self, action_type: str, target: str) -> bool:
        prompt = (
            f"[Guardian] HIGH-RISK {action_type}: {target[:160]}\n  Approve? [y/N]: "
        )
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, input, prompt)
            return response.strip().lower() in ("y", "yes")
        except Exception:  # noqa: BLE001
            return False  # default deny on any error

    async def _audit(
        self, action_type: str, target: str, decision: GateDecision
    ) -> None:
        if self.audit_logger is None:
            return
        event = "governance_approved" if decision.approved else "governance_blocked"
        try:
            await self.audit_logger.record(
                event,
                {
                    "action_type": action_type,
                    "target": target,
                    "risk": decision.risk,
                    "reason": decision.reason,
                },
                agent="guardian",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("guardian_audit_failed", error=str(e))
