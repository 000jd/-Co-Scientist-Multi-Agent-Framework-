"""
GovernanceGate — approves or blocks irreversible swarm actions.

Kimi Work security model:
- Read actions: pass freely
- State-changing actions (write, send, delete): pause for human approval
  OR auto-approve if confidence above threshold

Integrated into SwarmOrchestrator.execute() after merge, before pool write.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel

# Note: MergedResult will be imported at runtime to avoid circular dependencies
# from co_scientist.agents.swarm_orchestrator import MergedResult

logger = logging.getLogger("governance_gate")


class ActionRisk(str, Enum):
    READ = "read"          # always approved
    LOW = "low"            # auto-approved
    MEDIUM = "medium"      # approved if confidence > threshold
    HIGH = "high"          # always requires human confirmation


class GateDecision(BaseModel):
    approved: bool
    risk: ActionRisk
    reason: str
    modified_result: Optional[Any] = None


class GovernanceGate:
    """
    Gates irreversible outputs before they write to the hypothesis pool.
    
    In auto mode: approves if epistemic_uncertainty < threshold.
    In interactive mode: prompts human via stdin (for CLI usage).
    """
    
    def __init__(
        self,
        auto_approve_threshold: float = 0.6,   # uncertainty below this → auto-approve
        interactive: bool = False,               # True = prompt human on HIGH risk
        audit_logger=None,
    ):
        self.auto_approve_threshold = auto_approve_threshold
        self.interactive = interactive
        self.audit_logger = audit_logger
        
    async def check(
        self,
        merged: Any,
        context: Dict[str, Any],
    ) -> Any:
        """
        Evaluate merged result risk and approve/block/modify.
        Returns (possibly filtered) MergedResult.
        """
        risk = self._assess_risk(merged, context)
        
        if risk == ActionRisk.READ:
            return merged  # pass-through
            
        if risk == ActionRisk.LOW:
            if self.audit_logger:
                await self.audit_logger.record("governance_approved", {"risk": risk.value, "auto": True})
            return merged
            
        if risk == ActionRisk.MEDIUM:
            avg_uncertainty = self._avg_uncertainty(merged)
            if avg_uncertainty < self.auto_approve_threshold:
                if self.audit_logger:
                    await self.audit_logger.record("governance_approved", {
                        "risk": risk.value, "uncertainty": avg_uncertainty, "auto": True
                    })
                return merged
            else:
                logger.warning(f"GovernanceGate: HIGH uncertainty {avg_uncertainty:.2f} — stripping uncertain hypotheses")
                merged = self._strip_high_uncertainty(merged, threshold=0.8)
                if self.audit_logger:
                    await self.audit_logger.record("governance_filtered", {
                        "risk": risk.value, "uncertainty": avg_uncertainty
                    })
                return merged
                
        # HIGH risk
        if self.interactive:
            approved = await self._prompt_human(merged, context)
            if not approved:
                if self.audit_logger:
                    await self.audit_logger.record("governance_blocked", {"risk": risk.value, "reason": "human_rejected"})
                # We can't easily return an empty MergedResult without importing it, so we import here.
                from co_scientist.agents.swarm_orchestrator import MergedResult
                return MergedResult()  # empty — nothing written to pool
                
        return merged
        
    def _assess_risk(self, merged: Any, context: Dict[str, Any]) -> ActionRisk:
        """Risk is HIGH if any hypothesis has extremely high uncertainty or flags."""
        if not merged.new_hypotheses:
            return ActionRisk.READ
            
        # Check for safety flags in context
        if context.get("safety_flags"):
            return ActionRisk.HIGH
            
        avg_u = self._avg_uncertainty(merged)
        if avg_u > 0.85:
            return ActionRisk.HIGH
        elif avg_u > 0.65:
            return ActionRisk.MEDIUM
        return ActionRisk.LOW
        
    def _avg_uncertainty(self, merged: Any) -> float:
        if not merged.new_hypotheses:
            return 0.0
        scores = []
        for h in merged.new_hypotheses:
            if isinstance(h, dict):
                candidates = h.get("candidates", [])
                for c in candidates:
                    u = c.get("epistemic_uncertainty_score", 0.5) if isinstance(c, dict) else getattr(c, "epistemic_uncertainty_score", 0.5)
                    scores.append(u)
            else:
                candidates = getattr(h, "candidates", [])
                for c in candidates:
                    u = getattr(c, "epistemic_uncertainty_score", 0.5)
                    scores.append(u)
        return sum(scores) / len(scores) if scores else 0.0
        
    def _strip_high_uncertainty(self, merged: Any, threshold: float) -> Any:
        """Remove hypotheses where all candidates have uncertainty > threshold."""
        filtered_hyps = []
        for h in merged.new_hypotheses:
            if isinstance(h, dict):
                candidates = h.get("candidates", [])
                max_u = max((c.get("epistemic_uncertainty_score", 0.5) if isinstance(c, dict) else getattr(c, "epistemic_uncertainty_score", 0.5) for c in candidates), default=0.5)
            else:
                candidates = getattr(h, "candidates", [])
                max_u = max((getattr(c, "epistemic_uncertainty_score", 0.5) for c in candidates), default=0.5)
                
            if max_u <= threshold:
                filtered_hyps.append(h)
        merged.new_hypotheses = filtered_hyps
        return merged
        
    async def _prompt_human(self, merged: Any, context: Dict[str, Any]) -> bool:
        """Interactive CLI prompt for HIGH-risk actions."""
        print(f"\n[GovernanceGate] HIGH RISK action requires approval.")
        print(f"  Cycle: {context.get('cycle')}")
        print(f"  New hypotheses: {len(merged.new_hypotheses)}")
        print(f"  Avg uncertainty: {self._avg_uncertainty(merged):.2f}")
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, input, "  Approve? [y/N]: ")
            return response.strip().lower() in ("y", "yes")
        except Exception:
            return False  # default deny on error
