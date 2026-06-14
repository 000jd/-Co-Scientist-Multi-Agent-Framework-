"""
Supervisor Agent for Domain-Agnostic Interventions.
"""

from typing import Dict, Any, List
from co_scientist.agents.base import BaseAgent
from co_scientist.core.hypothesis import Hypothesis

class SupervisorAgent(BaseAgent):
    """Pipeline supervisor. Manages Robin subsystem and governance gates."""
    agent_name = "supervisor"
    
    def __init__(self, config, event_bus, llm_router, tool_registry=None):
        super().__init__(config, event_bus, llm_router)
        self.tool_registry = tool_registry
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        hypotheses: List[Hypothesis] = context["hypotheses"]
        cycle = context.get("cycle", 0)
        convergence = context.get("convergence_tracker")

        # Check PlannerAgent convergence hint first (cheapest gate)
        swarm_plan = context.get("swarm_plan")
        if swarm_plan:
            hint = getattr(swarm_plan, "convergence_hint", "")
            if hint and "converged" in hint.lower():
                return {"action": "trigger_meta_review", "reason": "planner_convergence_hint"}
        
        # 1. Check governance gates
        gated = [h for h in hypotheses if any(
            getattr(c, "requires_governance_gate", False) for c in h.candidates
        )]
        
        # 2. Trigger Robin for top-K
        top_k = self.config.experiment.top_k_for_evolution
        top = sorted(hypotheses, key=lambda h: h.trueskill_rating.conservative_rating, reverse=True)[:top_k]
        
        robin_results = []
        import asyncio
        from co_scientist.agents.robin.albatross import AlbatrossAgent
        from co_scientist.agents.robin.condor import CondorAgent
        from co_scientist.agents.robin.osprey import OspreyAgent
        
        albatross = AlbatrossAgent(self.config, self.event_bus, self.llm, self.tool_registry)
        condor = CondorAgent(self.config, self.event_bus, self.llm, self.tool_registry)
        osprey = OspreyAgent(self.config, self.event_bus, self.llm, self.tool_registry)
        
        for h in top:
            ctx = {"hypothesis": h}
            res = await asyncio.gather(
                albatross.execute(ctx),
                condor.execute(ctx),
                osprey.execute(ctx),
                return_exceptions=True
            )
            robin_results.append({
                "hypothesis_id": h.id,
                "albatross": res[0] if not isinstance(res[0], Exception) else str(res[0]),
                "condor": res[1] if not isinstance(res[1], Exception) else str(res[1]),
                "osprey": res[2] if not isinstance(res[2], Exception) else str(res[2]),
            })
            
        # 3. Check for pending CDS data fetch jobs
        pending_jobs = await self._check_pending_fetches()
        if pending_jobs:
            return {"action": "wait_for_data", "pending_jobs": pending_jobs}
            
        # 4. Check convergence
        if convergence and convergence.is_converged:
            return {"action": "trigger_meta_review", "reason": "convergence"}
            
        # Also check if we're just spinning wheels (no new hypotheses last cycle)
        recent = [h for h in hypotheses if h.cycle_number >= cycle - 1]
        if len(recent) == 0 and cycle > 3:
            return {"action": "trigger_meta_review", "reason": "stagnation"}
            
        if self._is_converged(hypotheses, cycle):
            return {"action": "trigger_meta_review"}
            
        return {"action": "continue", "robin_results": robin_results, "gated": gated}
        
    async def _run_robin(self, hypothesis: Hypothesis) -> Dict[str, Any]:
        return {"hypothesis_id": hypothesis.id, "status": "completed"}
        
    async def _check_pending_fetches(self) -> List[Any]:
        return []
        
    def _is_converged(self, hypotheses: List[Hypothesis], cycle: int) -> bool:
        if cycle >= self.config.experiment.max_cycles:
            return True
        return False
