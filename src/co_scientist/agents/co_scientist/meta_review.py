"""
Meta-Review Agent for Domain-Agnostic Interventions.
"""

from typing import Dict, Any, List
from pydantic import BaseModel, Field
from co_scientist.agents.base import BaseAgent

class MetaReviewOutput(BaseModel):
    what_worked: List[str] = Field(description="Successful approaches in this cycle")
    what_failed: List[str] = Field(description="Failed or dead-end approaches in this cycle")
    recommended_focus: str = Field(description="Specific advice for the next generation cycle")
    report: str = Field(description="A comprehensive summary of the cycle")

class MetaReviewAgent(BaseAgent):
    """Produces final synthesis of the pipeline output."""
    agent_name = "meta_review"
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        leaderboard = context.get("leaderboard", [])
        hypotheses = context.get("hypotheses", [])
        cycle = context.get("cycle", "Final")
        
        prompt = f"""Synthesize the current state of the research pipeline for Cycle {cycle}.
Review the top hypotheses on the leaderboard and identify patterns, gaps, and promising directions.
Provide actionable insights that the Generation Agent can use in the next cycle to produce better, more novel hypotheses.

Top Hypotheses:
"""
        for entry in leaderboard[:5]:
            prompt += f"- {entry.get('title', 'Unknown')} (Score: {entry.get('mu', 0):.2f})\n"
            
        try:
            response = await self._call_llm(
                system_prompt="You are a Meta-Reviewer evaluating a research pipeline. Provide critical, actionable insights.",
                user_prompt=prompt,
                response_format=MetaReviewOutput
            )
            return response.model_dump()
        except Exception as e:
            import logging
            logging.getLogger("meta_review").warning(f"LLM Meta-Review failed: {e}")
            return {
                "what_worked": [],
                "what_failed": [],
                "recommended_focus": "No insights due to error",
                "report": "No insights available due to an error."
            }
