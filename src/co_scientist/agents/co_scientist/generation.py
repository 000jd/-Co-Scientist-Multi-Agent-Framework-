"""
Generation Agent for Domain-Agnostic Hypotheses.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from co_scientist.agents.base import BaseAgent
from co_scientist.core.domain_models import Candidate
from co_scientist.core.hypothesis import Hypothesis
from co_scientist.core.events import EventType
from co_scientist.tools.registry import ToolRegistry

class GeneratedCandidate(BaseModel):
    name: str
    description: str
    domain_type: str = "generic"
    metrics: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)

class GeneratedHypothesis(BaseModel):
    title: str
    summary: str
    rationale: str
    candidates: List[GeneratedCandidate]

class GenerationOutput(BaseModel):
    hypotheses: List[GeneratedHypothesis]

class GenerationAgent(BaseAgent):
    """Generates novel hypotheses based on a generic domain."""
    agent_name = "generation"
    
    def __init__(self, config, event_bus, llm_router, tool_registry: ToolRegistry = None):
        super().__init__(config, event_bus, llm_router)
        self.tool_registry = tool_registry or ToolRegistry()
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        research_goal = context["research_goal"]
        existing_hypotheses = context.get("existing_hypotheses", [])
        cycle = context.get("cycle", 0)
        previous_insights = context.get("previous_insights", "")
        scope = context.get("scope", "Generate hypotheses")
        
        # Construct LLM prompt
        system_prompt = self._build_system_prompt(self.config.domain.name, cycle, previous_insights)
        user_prompt = self._build_user_prompt(scope, research_goal, existing_hypotheses)
        
        # Call LLM with structured output
        response = await self._call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format=GenerationOutput,
        )
        
        # Create Hypothesis objects
        hypotheses = []
        for h_data in response.hypotheses:
            candidates = [
                Candidate(**c.model_dump()) for c in h_data.candidates
            ]
            hypothesis = Hypothesis[Candidate](
                research_goal=research_goal,
                title=h_data.title,
                summary=h_data.summary,
                rationale=h_data.rationale,
                candidates=candidates,
                source_agent="generation",
                cycle_number=cycle,
            )
            hypotheses.append(hypothesis)
            await self._publish_event(
                EventType.HYPOTHESIS_GENERATED,
                {"hypothesis_id": hypothesis.id, "hypothesis": hypothesis}
            )
            
        return {"hypotheses": hypotheses}
        
    def _build_system_prompt(self, domain_name: str, cycle: int, previous_insights: str = "") -> str:
        prompt = (
            f"You are an AI research assistant. Generate novel hypotheses in the domain: {domain_name}.\n"
            "Requirements:\n"
            "- Each hypothesis must have a name, description, and a set of metrics relevant to that domain.\n"
        )
        if previous_insights:
            prompt += f"\nKey insights from previous cycles to incorporate:\n{previous_insights}\n"
        return prompt
        
    def _build_user_prompt(self, scope: str, research_goal: str, existing_hypotheses: List[Hypothesis]) -> str:
        prompt = f"Scope for this task: {scope}\n\nOverall goal: {research_goal}\n"
        if existing_hypotheses:
            prompt += f"\nExisting hypotheses to build upon or differentiate from:\n"
            for h in existing_hypotheses[:5]:
                prompt += f"- {h.title}: {h.summary[:200]}\n"
        return prompt
