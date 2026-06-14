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
from co_scientist.agents.co_scientist.planner import PlannerAgent, PlannerOutput, SwarmTask as PlannerTask
from co_scientist.agents.co_scientist.worker_pool import WorkerPool, WorkerSpec, ResultMerger


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
        # PlannerAgent instantiated lazily (avoids circular imports at module load)
        self._planner: Optional[PlannerAgent] = None
        # WorkerPool for count-based fan-out (WorkerSpec.count > 1)
        max_concurrent = getattr(getattr(config, "agents", None), "swarm_max_workers", 5)
        self._worker_pool = WorkerPool(sub_agents, max_concurrent=max_concurrent)
        
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
        convergence_info = context.get("convergence_info", "")
        
        await self._publish_event(EventType.STAGE_STARTED, {"stage": "swarm_plan", "cycle": cycle})

        # Step 1: PlannerAgent plans the cycle (K2.6 with full pool context)
        plan = await self._plan_cycle(goal, pool, cycle, previous_insights, max_workers, convergence_info)
        self.logger.info(f"Swarm plan: {len(plan.tasks)} tasks, merge={plan.merge_strategy}")

        if self.audit_logger:
            await self.audit_logger.record("swarm_plan", {"cycle": cycle, "tasks": len(plan.tasks), "reasoning": plan.reasoning[:200]})

        await self._publish_event(EventType.STAGE_STARTED, {"stage": "swarm_fanout", "task_count": len(plan.tasks)})

        # Step 2: Build WorkerSpec list from plan tasks and fan-out via WorkerPool
        # WorkerPool gives per-worker error isolation + semaphore-bounded concurrency
        base_context = {
            "research_goal": goal,
            "cycle": cycle,
            "previous_insights": previous_insights,
            "existing_hypotheses": list(pool.hypotheses.values()),
            "all_hypotheses": list(pool.hypotheses.values()),
            "pool_summary": list(pool.hypotheses.values())[:20],
        }
        worker_specs = [
            WorkerSpec(
                agent_type=t.agent_type,
                count=1,            # PlannerOutput.tasks are 1-per-task today;
                subtask=t.scope,    # WorkerPool supports count>1 for future use
                priority=t.priority,
            )
            for t in plan.tasks
            if t.agent_type in self.sub_agents
        ]
        worker_results = await self._worker_pool.execute_plan(worker_specs, base_context)

        # Step 3: Apply injection guard to any web content
        for wr in worker_results:
            if not wr.failed and self.injection_guard and "web_content" in wr.result:
                wr.result["web_content"] = await self.injection_guard.sanitize(wr.result["web_content"])

        # Step 4: ResultMerger — typed merge back into HypothesisPool
        merge_summary = ResultMerger.merge(worker_results, pool)
        if merge_summary["errors"]:
            for err in merge_summary["errors"]:
                self.logger.warning(f"[WorkerPool error] {err}")

        # Build a MergedResult-compatible object for backward compat with orchestrator
        merged = MergedResult(
            new_hypotheses=[h.model_dump() if hasattr(h, "model_dump") else h
                            for h in merge_summary["new_hypotheses"]],
            search_findings=[],
            synthesis_notes="",
        )

        # Step 5: Governance gate
        if self.governance_gate:
            merged = await self.governance_gate.check(merged, context)

        await self._publish_event(EventType.STAGE_COMPLETED, {"stage": "swarm_cycle", "cycle": cycle})

        failed_count = sum(1 for wr in worker_results if wr.failed)
        return {
            "merged": merged,
            "plan": plan,
            "task_count": len(worker_specs),
            "failed_count": failed_count,
        }
        
    async def _plan_cycle(
        self,
        goal: str,
        pool: HypothesisPool,
        cycle: int,
        previous_insights: str,
        max_workers: int,
        convergence_info: str = "",
    ) -> SwarmPlan:
        """Delegate planning to PlannerAgent (K2.6 with top-30 long-context)."""
        # Lazy-init PlannerAgent to avoid circular imports at module load
        if self._planner is None:
            self._planner = PlannerAgent(self.config, self.event_bus, self.llm)
            if self.memory:
                self._planner.memory = self.memory

        long_context_top_n = getattr(getattr(self.config, "agents", None), "long_context_top_n", 30)
        planner_enabled = getattr(getattr(self.config, "agents", None), "planner_enabled", True)

        if planner_enabled:
            try:
                result = await self._planner.execute({
                    "pool": pool,
                    "leaderboard": [],  # no leaderboard at this point; ranking is post-swarm
                    "cycle": cycle,
                    "previous_insights": previous_insights,
                    "research_goal": goal,
                    "convergence_info": convergence_info,
                    "max_workers": max_workers,
                    "long_context_top_n": long_context_top_n,
                })
                planner_output: PlannerOutput = result["plan"]
                # Adapt PlannerOutput.tasks → SwarmPlan.tasks, filter to known agents
                valid_tasks = [
                    SwarmTask(
                        agent_type=t.agent_type,
                        scope=t.scope,
                        context_snippet=t.context_snippet,
                        priority=t.priority,
                    )
                    for t in planner_output.tasks
                    if t.agent_type in self.sub_agents
                ][:max_workers]

                if valid_tasks:
                    return SwarmPlan(
                        reasoning=planner_output.reasoning,
                        tasks=valid_tasks,
                        merge_strategy=planner_output.merge_strategy,
                    )
                self.logger.warning("PlannerAgent returned no valid tasks, falling back to default plan")
            except Exception as e:
                self.logger.warning(f"PlannerAgent failed ({e}), falling back to default plan")

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
