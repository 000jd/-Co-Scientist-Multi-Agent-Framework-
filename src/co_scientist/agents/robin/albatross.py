"""Albatross: deep literature synthesis for a single hypothesis."""
from typing import Dict, Any
from co_scientist.agents.base import BaseAgent
from co_scientist.core.hypothesis import Hypothesis, Evidence

class AlbatrossAgent(BaseAgent):
    agent_name = "robin_albatross"
    
    def __init__(self, config, event_bus, llm_router, tool_registry=None):
        super().__init__(config, event_bus, llm_router)
        self.tool_registry = tool_registry
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        hypothesis: Hypothesis = context["hypothesis"]
        query = f"{hypothesis.title} {hypothesis.summary} evidence mechanism"
        if self.tool_registry and "literature_search" in self.tool_registry.available_tools:
            res = await self.tool_registry.call(
                "literature_search", 
                query=query, 
                max_results=20, 
                sources=["semantic_scholar", "arxiv", "crossref"]
            )
            papers = res.data if res.success else []
        else:
            from co_scientist.search.literature_search import LiteratureSearcher
            searcher = LiteratureSearcher(config={})
            papers = await searcher.search(query, max_results=20, sources=["semantic_scholar", "arxiv", "crossref"])
        
        if not papers:
            return {"hypothesis_id": hypothesis.id, "status": "no_literature_found", "papers": []}
            
        # Ask K2.6 to synthesize the literature
        papers_text = "\n".join([
            f"- {p.get('title', '')} ({p.get('year', '')}) [{p.get('source', '')}]: {p.get('abstract', '')[:200]}"
            for p in papers[:10]
        ])
        
        synthesis = await self._call_llm(
            system_prompt="You are a literature synthesis expert. Be objective and cite uncertainty.",
            user_prompt=f"Synthesize this literature in relation to the hypothesis:\n\nHypothesis: {hypothesis.title}\n{hypothesis.summary}\n\nPapers:\n{papers_text}\n\nDoes the literature support, refute, or is neutral? What is the confidence level?"
        )
        
        hypothesis.add_evidence(Evidence(
            source="albatross_literature",
            evidence_type="literature_synthesis",
            details=str(synthesis)[:2000],
            confidence=0.8,
            metadata={"papers_found": len(papers)},
        ))
        
        return {
            "hypothesis_id": hypothesis.id,
            "status": "completed",
            "papers_found": len(papers),
            "synthesis": str(synthesis)[:500],
        }
