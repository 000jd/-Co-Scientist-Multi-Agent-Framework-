"""
SwarmOrchestrator — Kimi Work lead-orchestrator pattern.

Decomposes a research cycle into parallel scoped sub-agent tasks.
All sub-agents are dispatched concurrently via asyncio.gather().
Results are merged at a synthesis gate before being written to the pool.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field
import logging

from co_scientist.agents.base import BaseAgent
from co_scientist.core.config import Config
from co_scientist.core.events import EventBus, EventType, Event
from co_scientist.core.hypothesis import Hypothesis, HypothesisPool
from co_scientist.infrastructure.llm import LLMRouter


class SwarmTask(BaseModel):
    """A scoped task delegated to one sub-agent worker."""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent_type: str          # "generation" | "search" | "reflection" | "evolution" | "validate"
    scope: str               # what specifically this worker should focus on
    context_snippet: str     # relevant slice of pool state for this task
    priority: int = 5        # 1=highest
    

class SwarmPlan(BaseModel):
    """K2.6 decides the cycle decomposition."""
    reasoning: str
    tasks: List[SwarmTask]
    merge_strategy: str      # "union" | "vote" | "best_score"
    

class MergedResult(BaseModel):
    new_hypotheses: List[Dict[str, Any]] = Field(default_factory=list)
    search_findings: List[Dict[str, Any]] = Field(default_factory=list)
    criticisms: List[Dict[str, Any]] = Field(default_factory=list)
    synthesis_notes: str = ""


class SwarmOrchestrator(BaseAgent):
    """
    Lead orchestrator implementing Kimi Work's manager-not-commander delegation.
    
    Pattern:
    1. Receive cycle context (goal + pool state + previous insights)
    2. Ask K2.6 to produce a SwarmPlan (what agents to spawn, what scope each gets)
    3. asyncio.gather() all sub-agent coroutines (gated by LLMRouter semaphore)
    4. Merge results at synthesis gate
    5. Apply GovernanceGate for irreversible actions
    6. Return merged pool delta
    """
    agent_name = "swarm_orchestrator"
    
    def __init__(
        self,
        config: Config,
        event_bus: EventBus,
        llm_router: LLMRouter,
        sub_agents: Dict[str, BaseAgent],
        governance_gate: Optional[Any] = None,
        audit_logger: Optional[Any] = None,
        injection_guard: Optional[Any] = None,
    ):
        super().__init__(config, event_bus, llm_router)
        self.sub_agents = sub_agents            # {"generation": GenerationAgent, ...}
        self.governance_gate = governance_gate
        self.audit_logger = audit_logger
        self.injection_guard = injection_guard
        self.logger = logging.getLogger("swarm_orchestrator")
        
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        context keys:
          research_goal: str
          pool: HypothesisPool
          cycle: int
          previous_insights: str
          max_workers: int  (default 5, tune down for free-tier K2.6)
        """
        goal = context["research_goal"]
        pool: HypothesisPool = context["pool"]
        cycle = context.get("cycle", 1)
        previous_insights = context.get("previous_insights", "")
        max_workers = context.get("max_workers", 5)
        
        await self._publish_event(EventType.STAGE_STARTED, {"stage": "swarm_plan", "cycle": cycle})
        
        # Step 1: K2.6 plans the cycle
        plan = await self._plan_cycle(goal, pool, cycle, previous_insights, max_workers)
        self.logger.info(f"Swarm plan: {len(plan.tasks)} tasks, merge={plan.merge_strategy}")
        
        if self.audit_logger:
            await self.audit_logger.record("swarm_plan", {"cycle": cycle, "tasks": len(plan.tasks), "reasoning": plan.reasoning[:200]})
            
        # Step 2: Fan-out — dispatch all tasks concurrently
        await self._publish_event(EventType.STAGE_STARTED, {"stage": "swarm_fanout", "task_count": len(plan.tasks)})
        
        task_coroutines = []
        for task in plan.tasks:
            coro = self._dispatch_task(task, goal, pool, cycle, previous_insights)
            task_coroutines.append(coro)
            
        # asyncio.gather — all sub-agents run in parallel, semaphore gates LLM calls
        raw_results = await asyncio.gather(*task_coroutines, return_exceptions=True)
        
        # Step 3: Filter exceptions, apply injection guard
        valid_results = []
        for task, result in zip(plan.tasks, raw_results):
            if isinstance(result, Exception):
                self.logger.warning(f"Task {task.task_id} ({task.agent_type}) failed: {result}")
                continue
            if self.injection_guard and "web_content" in result:
                result["web_content"] = await self.injection_guard.sanitize(result["web_content"])
            valid_results.append((task, result))
            
        # Step 4: Merge gate
        merged = await self._merge_results(valid_results, plan.merge_strategy, goal)
        
        # Step 5: Governance gate for any irreversible outputs
        if self.governance_gate:
            merged = await self.governance_gate.check(merged, context)
            
        await self._publish_event(EventType.STAGE_COMPLETED, {"stage": "swarm_cycle", "cycle": cycle})
        
        return {
            "merged": merged,
            "plan": plan,
            "task_count": len(plan.tasks),
            "failed_count": len(plan.tasks) - len(valid_results),
        }
        
    async def _plan_cycle(
        self,
        goal: str,
        pool: HypothesisPool,
        cycle: int,
        previous_insights: str,
        max_workers: int,
    ) -> SwarmPlan:
        """Ask K2.6 to decompose this cycle into parallel sub-tasks."""
        
        # Serialize top-10 pool state into prompt (exploit 256K ctx)
        top_hyps = sorted(
            pool.hypotheses.values(),
            key=lambda h: h.trueskill_rating.conservative_rating,
            reverse=True
        )[:10]
        
        pool_summary = "\n".join([
            f"- [{h.id}] {h.title} (score={h.trueskill_rating.conservative_rating:.1f}, "
            f"criticisms={len(h.criticisms)}, evidence={len(h.evidence)})"
            for h in top_hyps
        ])
        
        prompt = f"""You are the lead orchestrator for a self-improving scientific discovery system.

Research Goal: {goal}
Cycle: {cycle}
Max parallel workers: {max_workers}

Current top hypotheses:
{pool_summary if pool_summary else "[No hypotheses yet — this is cycle 1]"}

Previous cycle insights:
{previous_insights[:2000] if previous_insights else "[None — first cycle]"}

Your job: decompose this cycle into {min(max_workers, 5)} parallel sub-agent tasks.

Available agent types: generation, search, reflection, evolution

Rules:
- generation tasks: create NEW hypotheses that fill gaps or explore new directions
- search tasks: find evidence for specific hypotheses or open questions
- reflection tasks: critique specific hypotheses by ID
- evolution tasks: refine top-ranked hypotheses that have criticisms

Each task gets a "scope" (what to focus on) and a "context_snippet" (relevant hypothesis IDs/text).
Set merge_strategy to "best_score" if tasks produce competing hypotheses, "union" if they're complementary.

CRITICAL: Do NOT spawn more than {max_workers} tasks. Quality over quantity."""

        try:
            plan = await self._call_llm(
                system_prompt="You are a precise research orchestrator. Return a valid JSON SwarmPlan.",
                user_prompt=prompt,
                response_format=SwarmPlan,
            )
            plan.tasks = [t for t in plan.tasks if t.agent_type in self.sub_agents][:max_workers]
            return plan
        except Exception as e:
            self.logger.warning(f"SwarmPlan LLM call failed ({e}), using default plan")
            return self._default_plan(goal, pool, cycle, max_workers)
            
    def _default_plan(self, goal: str, pool: HypothesisPool, cycle: int, max_workers: int) -> SwarmPlan:
        """Fallback plan if K2.6 planning call fails."""
        tasks = [
            SwarmTask(agent_type="generation", scope=f"Generate 3 novel hypotheses for: {goal}", context_snippet=""),
            SwarmTask(agent_type="search", scope=f"Find recent evidence for: {goal}", context_snippet=""),
        ]
        if cycle > 1 and pool.hypotheses:
            tasks.append(SwarmTask(
                agent_type="reflection",
                scope="Critique top 3 hypotheses",
                context_snippet=", ".join(list(pool.hypotheses.keys())[:3]),
            ))
            tasks.append(SwarmTask(
                agent_type="evolution",
                scope="Refine hypotheses addressing criticisms",
                context_snippet=", ".join(list(pool.hypotheses.keys())[:3]),
            ))
        return SwarmPlan(reasoning="Default fallback plan", tasks=tasks[:max_workers], merge_strategy="union")
        
    async def _dispatch_task(
        self,
        task: SwarmTask,
        goal: str,
        pool: HypothesisPool,
        cycle: int,
        previous_insights: str,
    ) -> Dict[str, Any]:
        """Route a SwarmTask to the appropriate sub-agent."""
        agent = self.sub_agents.get(task.agent_type)
        if not agent:
            raise ValueError(f"No agent registered for type: {task.agent_type}")
            
        # Build scoped context for this worker
        context = {
            "research_goal": goal,
            "cycle": cycle,
            "previous_insights": previous_insights,
            "scope": task.scope,
            "task_id": task.task_id,
        }
        
        if task.agent_type == "generation":
            context["existing_hypotheses"] = list(pool.hypotheses.values())
            
        elif task.agent_type in ("reflection", "evolution"):
            # Give agent only the hypotheses relevant to its scope
            relevant_ids = [hid.strip() for hid in task.context_snippet.split(",") if hid.strip()]
            relevant_hyps = [pool.get(hid) for hid in relevant_ids if pool.get(hid)]
            if not relevant_hyps:
                relevant_hyps = list(pool.hypotheses.values())[:5]
            context["hypotheses"] = relevant_hyps
            if task.agent_type == "evolution":
                context["top_hypotheses"] = relevant_hyps
                context["all_hypotheses"] = list(pool.hypotheses.values())
                
        elif task.agent_type == "search":
            context["query"] = f"{task.scope} {goal}"
            context["max_iterations"] = 2  # keep each search worker lean
            
        elif task.agent_type == "validate":
            relevant_ids = [hid.strip() for hid in task.context_snippet.split(",") if hid.strip()]
            relevant_hyps = [pool.get(hid) for hid in relevant_ids if pool.get(hid)]
            context["hypotheses"] = relevant_hyps or list(pool.hypotheses.values())[:3]
            
        if self.audit_logger:
            await self.audit_logger.record("task_dispatch", {
                "task_id": task.task_id,
                "agent_type": task.agent_type,
                "scope": task.scope[:100],
            })
            
        result = await agent.execute(context)
        
        if self.audit_logger:
            await self.audit_logger.record("task_complete", {
                "task_id": task.task_id,
                "agent_type": task.agent_type,
                "output_keys": list(result.keys()),
            })
            
        return result
        
    async def _merge_results(
        self,
        valid_results: List[tuple],
        merge_strategy: str,
        goal: str,
    ) -> MergedResult:
        """Collect and merge all sub-agent outputs into a unified delta."""
        merged = MergedResult()
        
        for task, result in valid_results:
            if task.agent_type == "generation":
                for h in result.get("hypotheses", []):
                    merged.new_hypotheses.append(h.model_dump() if hasattr(h, "model_dump") else h)
                    
            elif task.agent_type == "search":
                findings = result.get("sources", [])
                synthesis = result.get("synthesis", "")
                merged.search_findings.extend(findings)
                if synthesis:
                    merged.synthesis_notes += f"\n[Search: {task.scope[:50]}]\n{synthesis}\n"
                    
            elif task.agent_type == "reflection":
                for h in result.get("hypotheses", []):
                    for c in (h.criticisms if hasattr(h, "criticisms") else []):
                        merged.criticisms.append(c.model_dump() if hasattr(c, "model_dump") else c)
                        
            elif task.agent_type == "evolution":
                for h in result.get("evolved_hypotheses", []):
                    merged.new_hypotheses.append(h.model_dump() if hasattr(h, "model_dump") else h)
                    
        return merged
