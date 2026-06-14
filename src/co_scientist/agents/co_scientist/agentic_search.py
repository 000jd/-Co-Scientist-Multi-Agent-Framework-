from typing import Dict, Any, List
from pydantic import BaseModel, Field

from co_scientist.agents.base import BaseAgent
from co_scientist.search.web_search import WebSearchEngine, WebSearchResult


class SearchPlan(BaseModel):
    """The LLM decides what to search for next."""
    reasoning: str
    search_queries: List[str] = Field(default_factory=list)
    done: bool = False


class SearchMemory(BaseModel):
    """Accumulated knowledge from the search loop."""
    query: str
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    synthesis: str = ""


class AgenticSearchAgent(BaseAgent):
    """
    Autonomous research agent that searches the open web iteratively.
    """
    agent_name = "agentic_search"
    
    def __init__(self, config, event_bus, llm_router, tool_registry=None, injection_guard=None):
        super().__init__(config, event_bus, llm_router)
        # Pull API key from your existing SearchConfig
        api_key = getattr(config.search, "serper_api_key", None) or ""
        self.engine = WebSearchEngine(api_key=api_key) if api_key else None
        
        tinyfish_api_key = getattr(config.search, "tinyfish_api_key", None) or ""
        if tinyfish_api_key:
            from co_scientist.search.hybrid_search import HybridSearchEngine
            self.hybrid_engine = HybridSearchEngine(api_key=tinyfish_api_key)
        else:
            self.hybrid_engine = None
            
        self.tool_registry = tool_registry
        self.injection_guard = injection_guard
        
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Context expects:
        - query: str (the research question)
        - max_iterations: int (default 3)
        - hypothesis: Hypothesis (optional, to ground the search)
        """
        if not self.engine and not self.hybrid_engine:
            return {"error": "No web search API key configured", "synthesis": "No API Key", "sources": [], "hypothesis_id": ""}

        # Support swarm-injected hypothesis target: if no explicit query but pool_summary given,
        # pick the first hypothesis without evidence to ground this search worker.
        if "query" not in context and "pool_summary" in context:
            for h in context.get("pool_summary", []):
                if not getattr(h, "evidence", None):
                    context["query"] = f"{h.title} {h.summary[:150]}"
                    context["hypothesis_id"] = h.id
                    break

        if "query" not in context:
            return {"error": "No query provided", "synthesis": "", "sources": [], "hypothesis_id": ""}

        original_query = context["query"]
        max_iterations = context.get("max_iterations", 3)
        memory = SearchMemory(query=original_query)
        
        for i in range(max_iterations):
            # 1. Plan what to search based on what we know so far
            plan = await self._plan_next_search(original_query, memory, i)
            
            # Browser bridge starts and stops at orchestrator level now
            if plan.done or not plan.search_queries:
                break
                
            # 2. Execute searches
            for q in plan.search_queries:
                if self.engine:
                    try:
                        results = await self.engine.search(q, max_results=5)
                        
                        # 3. Read top pages
                        for r in results[:3]:
                            if self.tool_registry and "browser_navigate" in self.tool_registry.available_tools:
                                browse_res = await self.tool_registry.call("browser_navigate", url=r.url)
                                if browse_res.success:
                                    content = browse_res.data
                                    if self.injection_guard:
                                        safe_content = await self.injection_guard.sanitize(content, source_url=r.url)
                                    else:
                                        safe_content = content
                                    memory.findings.append({
                                        "search_query": q,
                                        "title": r.title,
                                        "url": r.url,
                                        "snippet": r.snippet,
                                        "content": safe_content,
                                    })
                                else:
                                    memory.findings.append({
                                        "search_query": q,
                                        "title": r.title,
                                        "url": r.url,
                                        "snippet": r.snippet,
                                        "content": f"[Failed to fetch via Tool Registry: {browse_res.error}]",
                                    })
                            else:
                                memory.findings.append({
                                    "search_query": q,
                                    "title": r.title,
                                    "url": r.url,
                                    "snippet": r.snippet,
                                    "content": "[No browser tool available]",
                                })
                    except Exception as e:
                        import logging
                        logging.getLogger("agentic_search").warning(f"Serper search failed, trying hybrid fallback: {e}")
                        if self.hybrid_engine:
                            await self._run_hybrid_fallback(q, memory)
                elif self.hybrid_engine:
                    await self._run_hybrid_fallback(q, memory)
                    
        # 4. Final synthesis
        synthesis = await self._synthesize(original_query, memory)
        memory.synthesis = synthesis
        
        # Browser bridge stops at orchestrator level
        
        return {
            "synthesis": synthesis,
            "sources": memory.findings,
            "memory": memory,
            "hypothesis_id": context.get("hypothesis_id", ""),
        }
        
    async def _run_hybrid_fallback(self, query: str, memory: SearchMemory):
        results = await self.hybrid_engine.search_and_fetch(query, max_results=3)
        for r in results:
            memory.findings.append({
                "search_query": query,
                "title": r.title,
                "url": r.url,
                "snippet": r.snippet[:500],
                "content": r.snippet,
            })
            
    async def _plan_next_search(
        self, 
        original_query: str, 
        memory: SearchMemory, 
        iteration: int
    ) -> SearchPlan:
        """Ask the LLM what to search for next based on accumulated findings."""
        
        findings_text = "\n\n".join([
            f"Source: {f['url']}\nContent: {f['content'][:500]}..."
            for f in memory.findings[-5:]  # Last 5 findings for context window
        ])
        
        prompt = f"""You are an autonomous research planner. Your goal: "{original_query}"

Iteration: {iteration + 1}

Current findings:
{findings_text if findings_text else "[No findings yet]"}

Decide:
1. Are we done? (set done=true if satisfied)
2. If not done, what specific search queries should we run next to fill gaps?

Be specific. Instead of "general data", use "specific dataset version 2024"."""

        try:
            response = await self._call_llm(
                system_prompt="You are a precise research strategist. Return JSON.",
                user_prompt=prompt,
                response_format=SearchPlan,
            )
            return response
        except Exception:
            # Fallback: just search the original query with a tweak
            return SearchPlan(
                reasoning="Fallback search",
                search_queries=[f"{original_query} latest research 2024"],
                done=False,
            )
            
    async def _synthesize(self, query: str, memory: SearchMemory) -> str:
        """Generate final research summary from all gathered sources."""
        sources_text = "\n\n".join([
            f"URL: {f['url']}\n{f['content'][:1000]}"
            for f in memory.findings
        ])
        
        prompt = f"""Synthesize the following web research into a concise, factual report.

Research Question: {query}

Sources:
{sources_text}

Provide:
1. Key findings (bullet points)
2. Confidence level for each
3. Gaps remaining
4. Recommended next steps"""

        try:
            return await self._call_llm(
                system_prompt="You are a research analyst. Be factual, cite uncertainty.",
                user_prompt=prompt,
            )
        except Exception:
            return "Synthesis failed due to LLM error."
