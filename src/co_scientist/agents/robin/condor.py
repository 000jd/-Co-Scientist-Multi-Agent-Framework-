"""Condor: multi-angle consensus check — does the hypothesis agree with domain consensus?"""
from typing import Dict, Any, List
from pydantic import BaseModel
from co_scientist.agents.base import BaseAgent
from co_scientist.core.hypothesis import Hypothesis, Criticism

class ConsensusVote(BaseModel):
    verdict: str       # "supports" | "neutral" | "contradicts"
    confidence: float
    rationale: str

class CondorAgent(BaseAgent):
    agent_name = "robin_condor"
    
    def __init__(self, config, event_bus, llm_router, tool_registry=None):
        super().__init__(config, event_bus, llm_router)
        self.tool_registry = tool_registry
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        hypothesis: Hypothesis = context["hypothesis"]
        
        # Run 3 independent LLM "reviewers" concurrently
        import asyncio
        perspectives = ["mechanistic plausibility", "empirical evidence", "theoretical consistency"]
        
        review_coros = [
            self._review_perspective(hypothesis, p) for p in perspectives
        ]
        votes: List[ConsensusVote] = await asyncio.gather(*review_coros, return_exceptions=True)
        
        valid_votes = [v for v in votes if isinstance(v, ConsensusVote)]
        
        if not valid_votes:
            return {"hypothesis_id": hypothesis.id, "status": "review_failed"}
            
        # Majority verdict
        supports = sum(1 for v in valid_votes if v.verdict == "supports")
        contradicts = sum(1 for v in valid_votes if v.verdict == "contradicts")
        
        if contradicts > supports:
            hypothesis.add_criticism(Criticism(
                critic_agent="condor",
                dimension="consensus",
                severity="medium",
                content=f"Multi-reviewer consensus check: {contradicts}/{len(valid_votes)} reviewers found contradictions",
                suggested_fix="Revise mechanism or add qualifying conditions",
            ))
            
        return {
            "hypothesis_id": hypothesis.id,
            "status": "completed",
            "supports": supports,
            "contradicts": contradicts,
            "votes": [v.model_dump() for v in valid_votes],
        }
        
    async def _review_perspective(self, hypothesis: Hypothesis, perspective: str) -> ConsensusVote:
        return await self._call_llm(
            system_prompt=f"You are a domain expert reviewing from the perspective of: {perspective}. Be critical.",
            user_prompt=f"Does this hypothesis align with current scientific consensus?\n\nTitle: {hypothesis.title}\nSummary: {hypothesis.summary}\n\nVerdict: supports/neutral/contradicts. Confidence 0-1. Short rationale.",
            response_format=ConsensusVote,
        )
