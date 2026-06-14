"""
PlannerAgent: K2.6-powered swarm planner.

Replaces the inline LLM call in SwarmOrchestrator._plan_cycle() with a
proper BaseAgent subclass. Reads full pool state (top-30 with criticisms
and evidence) and produces a SwarmPlan that decides which agents to run,
how many, and what their scoped subtasks are.

Wire-up: call PlannerAgent.execute() from SwarmOrchestrator._plan_cycle().
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from co_scientist.agents.base import BaseAgent
from co_scientist.core.hypothesis import HypothesisPool

logger = logging.getLogger("agent.planner")


class SwarmTask(BaseModel):
    """One scoped task for a single sub-agent worker."""
    agent_type: str       # "generation" | "reflection" | "evolution" | "search" | "validate"
    scope: str            # what this specific worker focuses on
    context_snippet: str  # relevant hypothesis IDs or text slice
    priority: int = 5     # 1 = highest


class PlannerOutput(BaseModel):
    """Full decomposition for one swarm cycle."""
    reasoning: str
    tasks: List[SwarmTask]
    merge_strategy: str = "union"   # "union" | "best_score"
    convergence_hint: str = ""      # consumed by SupervisorAgent
    meta_review_depth: str = "standard"  # "standard" | "deep" | "skip"


class PlannerAgent(BaseAgent):
    """
    LLM-driven cycle planner.

    Distinct from SwarmOrchestrator (which is the fan-out executor). This
    agent owns *what* to run; the SwarmOrchestrator owns *how* to run it.

    Uses K2.6's 256K context to read top-N hypotheses with their full
    criticisms and evidence, then decides agent allocation surgically.
    """
    agent_name = "planner"

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        pool: HypothesisPool = context["pool"]
        leaderboard: List[Dict] = context.get("leaderboard", [])
        cycle: int = context.get("cycle", 1)
        previous_insights: str = context.get("previous_insights", "")
        research_goal: str = context.get("research_goal", "")
        convergence_info: str = context.get("convergence_info", "")
        max_workers: int = context.get("max_workers", 5)
        top_n: int = context.get("long_context_top_n", 30)

        pool_summary = self._serialize_pool(pool, top_n=top_n)

        system_prompt = (
            "You are a master research orchestration planner. "
            "You have complete visibility into the current hypothesis pool and must decide "
            "the optimal swarm task decomposition for the next research cycle. "
            "Think like a research director: deploy workers where evidence gaps are, "
            "prune effort where hypotheses are already well-supported. "
            f"Maximum parallel workers available: {max_workers}. "
            "Do NOT spawn more tasks than max_workers. Quality over quantity. "
            "Return a valid JSON PlannerOutput."
        )

        user_prompt = f"""Research Goal: {research_goal}
Cycle: {cycle}

Previous Meta-Review Insights:
{previous_insights or '[First cycle — no prior insights]'}

Convergence Status:
{convergence_info or '[Not yet assessed]'}

Current Leaderboard (top 10):
{self._format_leaderboard(leaderboard[:10])}

Full Hypothesis Pool State (top {top_n}):
{pool_summary}

Based on the above, produce a PlannerOutput that:
1. Assigns tasks to agents: generation / reflection / evolution / search
2. Sets a scope per task (what specifically each worker focuses on)
3. Does NOT exceed {max_workers} total tasks
4. Flags convergence_hint if the leaderboard looks stable
5. Sets merge_strategy to "best_score" if tasks produce competing hypotheses, "union" if complementary

Rules per agent type:
- generation: create NEW hypotheses filling gaps or new directions
- search: find evidence for a specific open hypothesis (set context_snippet to hypothesis ID)
- reflection: critique specific hypotheses by ID (set context_snippet)
- evolution: refine top-ranked hypotheses that have unresolved criticisms

Be surgical. Don't just max out all agents."""

        try:
            plan: PlannerOutput = await self._call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format=PlannerOutput,
            )
            return {"plan": plan}
        except Exception as e:
            logger.warning(f"PlannerAgent LLM call failed ({e}), returning default plan")
            return {"plan": self._default_plan(research_goal, pool, cycle, max_workers)}

    def _serialize_pool(self, pool: HypothesisPool, top_n: int = 30) -> str:
        """Serialize top-N hypotheses with criticisms and evidence for long-context planning."""
        sorted_hyps = sorted(
            pool.hypotheses.values(),
            key=lambda h: h.trueskill_rating.conservative_rating,
            reverse=True,
        )[:top_n]

        if not sorted_hyps:
            return "[Pool is empty — cycle 1]"

        lines = []
        for h in sorted_hyps:
            crits = "; ".join(
                f"[{c.severity}] {c.dimension}: {c.content[:80]}"
                for c in h.criticisms[:3]
            )
            evid = "; ".join(
                f"{e.source}({e.confidence:.2f})"
                for e in h.evidence[:3]
            )
            lines.append(
                f"[{h.id}] {h.title}\n"
                f"  score={h.trueskill_rating.conservative_rating:.2f} "
                f"W/L={h.debate_wins}/{h.debate_losses} "
                f"status={h.status.value}\n"
                f"  summary: {h.summary[:120]}\n"
                f"  criticisms: {crits or 'none'}\n"
                f"  evidence: {evid or 'none'}"
            )
        return "\n\n".join(lines)

    def _format_leaderboard(self, lb: List[Dict]) -> str:
        if not lb:
            return "[Empty]"
        return "\n".join(
            f"{e.get('rank', i+1)}. [{e['id']}] {e['title'][:60]} "
            f"score={e.get('adjusted_score', e.get('elo', 0)):.2f} "
            f"flag={e.get('flag', e.get('tier', 'NOMINAL'))}"
            for i, e in enumerate(lb)
        )

    def _default_plan(
        self,
        goal: str,
        pool: HypothesisPool,
        cycle: int,
        max_workers: int,
    ) -> PlannerOutput:
        """Fallback if LLM planning call fails."""
        tasks = [
            SwarmTask(
                agent_type="generation",
                scope=f"Generate 3 novel hypotheses for: {goal}",
                context_snippet="",
            ),
            SwarmTask(
                agent_type="search",
                scope=f"Find recent evidence for: {goal}",
                context_snippet="",
            ),
        ]
        if cycle > 1 and pool.hypotheses:
            top_ids = ", ".join(list(pool.hypotheses.keys())[:3])
            tasks.append(SwarmTask(
                agent_type="reflection",
                scope="Critique top 3 hypotheses",
                context_snippet=top_ids,
            ))
            tasks.append(SwarmTask(
                agent_type="evolution",
                scope="Refine hypotheses addressing criticisms",
                context_snippet=top_ids,
            ))
        return PlannerOutput(
            reasoning="Default fallback plan (LLM call failed)",
            tasks=tasks[:max_workers],
            merge_strategy="union",
            convergence_hint="",
        )
