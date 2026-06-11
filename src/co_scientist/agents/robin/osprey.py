"""Osprey: designs a concrete, executable experiment to test the hypothesis."""
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from co_scientist.agents.base import BaseAgent
from co_scientist.core.hypothesis import Hypothesis

class ExperimentDesign(BaseModel):
    title: str
    objective: str
    methodology: str
    required_resources: List[str] = Field(default_factory=list)
    expected_outcomes: List[str] = Field(default_factory=list)
    failure_modes: List[str] = Field(default_factory=list)
    estimated_duration: str = ""
    cost_estimate: str = ""

class OspreyAgent(BaseAgent):
    agent_name = "robin_osprey"
    
    def __init__(self, config, event_bus, llm_router, tool_registry=None):
        super().__init__(config, event_bus, llm_router)
        self.tool_registry = tool_registry
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        hypothesis: Hypothesis = context["hypothesis"]
        
        # Testable predictions might not exist on the base model, so we need a fallback
        predictions = getattr(hypothesis, "testable_predictions", [])
        
        design = await self._call_llm(
            system_prompt="You are an experimental design expert. Design concrete, feasible experiments.",
            user_prompt=f"Design a minimal experiment to test this hypothesis:\n\nTitle: {hypothesis.title}\nSummary: {hypothesis.summary}\nPredictions: {predictions}\n\nBe specific. Include: title, objective, methodology, required_resources, expected_outcomes, failure_modes, estimated_duration, cost_estimate.",
            response_format=ExperimentDesign,
        )
        
        # Store design in hypothesis metadata if possible
        if hasattr(hypothesis, "testable_predictions"):
            hypothesis.testable_predictions.append(f"[Osprey Design] {design.objective}")
        
        return {
            "hypothesis_id": hypothesis.id,
            "status": "completed",
            "experiment": design.model_dump(),
        }
