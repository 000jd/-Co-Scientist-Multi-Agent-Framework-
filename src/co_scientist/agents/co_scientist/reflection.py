"""
Reflection Agent for Domain-Agnostic Hypotheses.
"""

from typing import Dict, Any, List
from pydantic import BaseModel
import importlib

from co_scientist.agents.base import BaseAgent
from co_scientist.core.hypothesis import Hypothesis, Criticism
from co_scientist.safety.base import SafetyVerdict

class ReflectionCritique(BaseModel):
    criticisms: List[Criticism]

class ReflectionAgent(BaseAgent):
    """Critiques hypotheses from multiple dimensions."""
    agent_name = "reflection"
    
    def __init__(self, config, event_bus, llm_router, safety_checker=None):
        super().__init__(config, event_bus, llm_router)
        self.safety_checker = safety_checker or self._load_safety_checker(config.domain.safety_module)
    
    def _load_safety_checker(self, module_path: str):
        if module_path == "default_safety":
            from co_scientist.safety.base import DefaultSafetyChecker
            return DefaultSafetyChecker()
        # Custom loading logic
        parts = module_path.split('.')
        module_name = '.'.join(parts[:-1])
        class_name = parts[-1]
        try:
            module = importlib.import_module(module_name)
            checker_class = getattr(module, class_name)
            return checker_class()
        except Exception as e:
            from co_scientist.safety.base import DefaultSafetyChecker
            return DefaultSafetyChecker()
            
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        hypotheses: List[Hypothesis] = context["hypotheses"]
        
        for hypothesis in hypotheses:
            for candidate in hypothesis.candidates:
                # Run safety assessment
                safety_report = await self.safety_checker.check(candidate, context)
                
                # Convert safety flags to criticisms
                for flag in safety_report.flags:
                    hypothesis.add_criticism(Criticism(
                        critic_agent="reflection_safety",
                        dimension="safety",
                        severity=flag.severity,
                        content=flag.description,
                        evidence=flag.evidence_basis,
                        suggested_fix=flag.recommended_action,
                    ))
                
                # Update epistemic uncertainty
                if hasattr(candidate, "increase_uncertainty"):
                    if safety_report.verdict in (SafetyVerdict.QUARANTINED, SafetyVerdict.REJECTED):
                        candidate.increase_uncertainty("safety_verdict_" + safety_report.verdict.value, 0.2)
                    elif safety_report.verdict == SafetyVerdict.CAUTION:
                        candidate.increase_uncertainty("safety_caution_flag", 0.1)
                
                # Multi-dimension LLM critique
                critique = await self._llm_critique(hypothesis, candidate)
                for c in critique.criticisms:
                    hypothesis.add_criticism(c)
        
        return {"hypotheses": hypotheses}
        
    async def _llm_critique(self, hypothesis: Hypothesis, candidate: Any) -> ReflectionCritique:
        prompt = (
            f"Critique this hypothesis on feasibility, safety, novelty, and impact in the domain: {self.config.domain.name}.\n"
            f"Title: {hypothesis.title}\n"
            f"Candidate: {candidate.name} - {candidate.description}"
        )
        response = await self._call_llm(
            system_prompt="You are a critical reviewer of scientific hypotheses.",
            user_prompt=prompt,
            response_format=ReflectionCritique
        )
        return response
