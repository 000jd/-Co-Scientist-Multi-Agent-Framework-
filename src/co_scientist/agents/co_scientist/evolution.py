"""
Evolution Agent for Domain-Agnostic Hypotheses.
"""

from typing import Dict, Any, List
import copy
from pydantic import BaseModel

from co_scientist.agents.base import BaseAgent
from co_scientist.core.hypothesis import Hypothesis, HypothesisStatus

class EvolutionOutput(BaseModel):
    title: str
    summary: str
    rationale: str
    candidates: List[Dict[str, Any]]

class EvolutionAgent(BaseAgent):
    """Evolves top hypotheses through combination and refinement."""
    agent_name = "evolution"
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        top_hypotheses: List[Hypothesis] = context["top_hypotheses"]
        
        evolved = []
        for hypothesis in top_hypotheses:
            # 1. Refine (address criticisms)
            refined = await self._evolve_refine(hypothesis)
            if refined:
                evolved.append(refined)
                
            # 2. Combine (if multiple available)
            if len(top_hypotheses) > 1:
                other = next((h for h in top_hypotheses if h.id != hypothesis.id), None)
                if other:
                    combined = await self._evolve_combine(hypothesis, other)
                    if combined:
                        evolved.append(combined)
                    
            # 3. Mutate
            mutated = await self._evolve_mutate(hypothesis)
            if mutated:
                evolved.append(mutated)
                
        for h in evolved:
            h.status = HypothesisStatus.EVOLVED
            h.generation = max(p.generation for p in top_hypotheses) + 1
            
        return {"evolved_hypotheses": evolved}
        
    async def _evolve_refine(self, hypothesis: Hypothesis) -> Hypothesis:
        """Refine hypothesis by addressing its criticisms via LLM."""
        from co_scientist.core.domain_models import Candidate
        criticisms_text = "\n".join([
            f"- [{c.severity}] {c.dimension}: {c.content}"
            for c in hypothesis.criticisms[:5]
        ])
        
        prompt = f"""Refine this hypothesis by addressing its criticisms.
    
Title: {hypothesis.title}
Summary: {hypothesis.summary}
Rationale: {hypothesis.rationale}

Criticisms to address:
{criticisms_text}

Generate an improved version with specific changes that address each criticism.
Return as JSON with: title, summary, rationale, candidates (list of dictionaries matching Candidate model)."""

        try:
            response = await self._call_llm(
                system_prompt="You are a research innovation expert. Refine hypotheses by addressing weaknesses.",
                user_prompt=prompt,
                response_format=EvolutionOutput,
            )
            
            new_h = Hypothesis[Candidate](
                parent_ids=[hypothesis.id],
                title=response.title,
                summary=response.summary,
                rationale=response.rationale,
                candidates=[Candidate(**c) for c in response.candidates],
                source_agent="evolution_refine",
                generation=hypothesis.generation + 1,
            )
            return new_h
        except Exception as e:
            import logging
            logging.getLogger("evolution").warning(f"LLM Refinement failed: {e}")
            return None

    async def _evolve_combine(self, h1: Hypothesis, h2: Hypothesis) -> Hypothesis:
        prompt = f"""Combine these two hypotheses into a single, stronger synthesis.
        
Hypothesis 1: {h1.title}
{h1.summary}

Hypothesis 2: {h2.title}
{h2.summary}

Return as JSON with: title, summary, rationale, candidates."""
        try:
            from co_scientist.core.domain_models import Candidate
            response = await self._call_llm(
                system_prompt="You are an expert synthesizer. Combine ideas synergistically.",
                user_prompt=prompt,
                response_format=EvolutionOutput,
            )
            new_h = Hypothesis[Candidate](
                parent_ids=[h1.id, h2.id],
                title=f"[Synthesis] {response.title}",
                summary=response.summary,
                rationale=response.rationale,
                candidates=[Candidate(**c) for c in response.candidates],
                source_agent="evolution_combine",
                generation=max(h1.generation, h2.generation) + 1,
            )
            return new_h
        except Exception as e:
            import logging
            logging.getLogger("evolution").warning(f"LLM Combine failed: {e}")
            return None
        
    async def _evolve_mutate(self, hypothesis: Hypothesis) -> Hypothesis:
        prompt = f"""Mutate this hypothesis to explore a different approach.
Change key parameters or core technology, while maintaining the same overall goal.

Title: {hypothesis.title}
Summary: {hypothesis.summary}
Rationale: {hypothesis.rationale}

Return as JSON with: title, summary, rationale, candidates (list of dictionaries matching Candidate model)."""
        try:
            from co_scientist.core.domain_models import Candidate
            response = await self._call_llm(
                system_prompt="You are a creative research expert. Mutate hypotheses into novel variants.",
                user_prompt=prompt,
                response_format=EvolutionOutput,
            )
            
            new_h = Hypothesis[Candidate](
                parent_ids=[hypothesis.id],
                title=f"[Mutated] {response.title}",
                summary=response.summary,
                rationale=response.rationale,
                candidates=[Candidate(**c) for c in response.candidates],
                source_agent="evolution_mutate",
                generation=hypothesis.generation + 1,
            )
            return new_h
        except Exception as e:
            import logging
            logging.getLogger("evolution").warning(f"LLM Mutation failed: {e}")
            return None
