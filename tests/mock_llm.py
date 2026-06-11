import json
import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional
from co_scientist.infrastructure.llm import BaseLLMProvider, Message, LLMResponse

class MockLLMProvider(BaseLLMProvider):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self.provider_name = "mock_provider"
        self._call_count = 0

    async def complete(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        self._call_count += 1
        
        # Analyze the prompt to decide what to return.
        prompt = messages[-1].content.lower() if messages else ""
        system = messages[0].content.lower() if messages and messages[0].role == "system" else ""
        
        content = ""
        
        if "research directions" in prompt or "research directions" in system:
            content = json.dumps({
                "directions": [{
                    "title": "Mock Direction",
                    "core_idea": "Use mock stuff",
                    "novelty_rationale": "It's mocked",
                    "proposed_mechanism": "Magic",
                    "potential_impact": "High"
                }]
            })
        elif "debate" in prompt or "debate" in system:
            if "winner" in prompt or "winner" in system or "criteria_scores" in prompt or "criteria_scores" in system:
                content = json.dumps({
                    "winner": "A",
                    "reasoning": "A is better",
                    "criteria_scores": {
                        "novelty": {"A": 8, "B": 7},
                        "plausibility": {"A": 8, "B": 7},
                        "testability": {"A": 8, "B": 7},
                        "impact": {"A": 8, "B": 7},
                        "argument_quality": {"A": 8, "B": 7}
                    },
                    "confidence": 0.9
                })
            elif "rebuttal" in prompt or "rebuttal" in system:
                content = "Mock Rebuttal"
            else:
                content = "Mock Argument"
        elif "scientific hypothesis" in prompt or "hypothesis" in system:
            # Hypothesis generation
            content = json.dumps({
                "title": "Mock Hypothesis",
                "statement": "Mock Statement",
                "mechanism": "Mock Mechanism",
                "predicted_outcome": "Mock Outcome",
                "testable_predictions": ["pred 1", "pred 2", "pred 3"],
                "novelty_score": 0.8,
                "plausibility_score": 0.8,
                "testability_score": 0.8,
                "impact_score": 0.8,
                "drug_candidates": [],
                "tags": ["mock"],
                "subfield": "mocking"
            })
        elif "literature review summary" in prompt or "synthesis" in prompt:
            content = json.dumps({
                "executive_summary": "Mock synthesis",
                "key_mechanisms": ["mech 1"],
                "critical_gaps": ["gap 1"],
                "emerging_opportunities": ["opp 1"]
            })
        else:
            # Default fallback JSON
            content = json.dumps({"status": "mocked", "message": "Default mock response"})

        return LLMResponse(
            content=content,
            model="mock_model",
            provider=self.provider_name,
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            latency_ms=1.0,
            parsed_json=json.loads(content) if "{" in content else None
        )

    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None
    ) -> List[List[float]]:
        # Return random 1536-dim vectors
        import random
        return [[random.random() for _ in range(1536)] for _ in texts]

    async def stream(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        yield "mocked"
        
class MockLiteratureSearcher:
    def __init__(self, config: Any):
        pass
        
    async def search(self, query: str, max_results: int = 10, sources: List[str] = None):
        return [
            {"title": "Fake Paper 1", "abstract": "Fake abstract 1", "authors": ["John Doe"]},
            {"title": "Fake Paper 2", "abstract": "Fake abstract 2", "authors": ["Jane Smith"]},
        ]
