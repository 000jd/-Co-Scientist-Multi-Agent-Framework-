"""
WorkerPool: asyncio-based swarm executor with count-based fan-out.

Takes a list of WorkerSpec (each with an optional count > 1) and fans out
agent.execute() coroutines in parallel. Implements the "up to N sub-agents"
pattern from Kimi Work's Agent Swarm.

WorkerResult and ResultMerger handle per-worker error isolation and
evidence/criticism attachment back to HypothesisPool.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from co_scientist.agents.base import BaseAgent
from co_scientist.core.hypothesis import Hypothesis, HypothesisPool

logger = logging.getLogger("worker_pool")


class WorkerSpec:
    """
    A fan-out specification: agent_type + how many workers + per-worker subtask.

    count > 1 spawns N identical workers, each receiving subtask + worker_idx
    in their context so they can differentiate (e.g. different search angles).
    """
    def __init__(
        self,
        agent_type: str,
        count: int = 1,
        subtask: str = "",
        priority: int = 1,
    ):
        self.agent_type = agent_type
        self.count = max(1, count)
        self.subtask = subtask
        self.priority = priority

    def __repr__(self) -> str:
        return f"WorkerSpec({self.agent_type}×{self.count})"


class WorkerResult:
    """Tagged result from one worker invocation."""
    def __init__(
        self,
        agent_type: str,
        worker_idx: int,
        result: Dict[str, Any],
        error: Optional[str] = None,
    ):
        self.agent_type = agent_type
        self.worker_idx = worker_idx
        self.result = result
        self.error = error

    @property
    def failed(self) -> bool:
        return self.error is not None


class WorkerPool:
    """
    Semaphore-bounded swarm executor.

    Fans out all WorkerSpec entries concurrently via asyncio.gather().
    Each entry with count=N spawns N concurrent workers, each getting
    (subtask, worker_idx) injected into context.

    Semaphore caps total concurrent LLM calls — safe for free-tier APIs.
    Per-worker exceptions are isolated: one failure doesn't abort the swarm.
    """

    def __init__(self, agent_registry: Dict[str, BaseAgent], max_concurrent: int = 10):
        self.agents = agent_registry
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_plan(
        self,
        worker_specs: List[WorkerSpec],
        base_context: Dict[str, Any],
    ) -> List[WorkerResult]:
        """
        Fan out all specs, return results tagged by (agent_type, worker_idx).
        Sorts by priority (1=highest) before launching.
        """
        sorted_specs = sorted(worker_specs, key=lambda s: s.priority)

        coros = []
        for spec in sorted_specs:
            for idx in range(spec.count):
                coros.append(self._run_worker(spec, idx, base_context))

        return list(await asyncio.gather(*coros, return_exceptions=False))

    async def _run_worker(
        self,
        spec: WorkerSpec,
        worker_idx: int,
        base_context: Dict[str, Any],
    ) -> WorkerResult:
        """Run one worker instance — semaphore-gated, exception-isolated."""
        async with self._semaphore:
            agent = self.agents.get(spec.agent_type)
            if agent is None:
                return WorkerResult(
                    spec.agent_type, worker_idx, {},
                    error=f"No agent registered for type '{spec.agent_type}'"
                )

            # Each worker gets its own shallow-copied context with injected fields
            ctx = {
                **base_context,
                "subtask": spec.subtask,
                "scope": spec.subtask,      # generation.py reads "scope" today
                "worker_idx": worker_idx,
            }

            try:
                label = f"{spec.agent_type}[{worker_idx}]"
                logger.info(f"[WorkerPool] {label} starting: {spec.subtask[:60]!r}")
                result = await agent.execute(ctx)
                logger.info(f"[WorkerPool] {label} done")
                return WorkerResult(spec.agent_type, worker_idx, result)
            except Exception as exc:
                logger.warning(f"[WorkerPool] {spec.agent_type}[{worker_idx}] failed: {exc}")
                return WorkerResult(spec.agent_type, worker_idx, {}, error=str(exc))


class ResultMerger:
    """
    Merges WorkerPool results back into the shared HypothesisPool.

    Agent type → what it contributes:
      generation  → new Hypothesis objects → pool.add()
      reflection  → Criticism objects attached to existing hypotheses
      evolution   → evolved Hypothesis objects → pool.add()
      search      → Evidence attached to the targeted hypothesis
      validation  → (no-op for now; validators update status directly)

    Returns a summary dict for logging/telemetry.
    """

    @staticmethod
    def merge(
        results: List[WorkerResult],
        pool: HypothesisPool,
    ) -> Dict[str, Any]:
        from co_scientist.core.hypothesis import Evidence

        new_hypotheses: List[Hypothesis] = []
        total_criticisms = 0
        total_evidence = 0
        errors: List[str] = []

        for wr in results:
            if wr.failed:
                errors.append(f"{wr.agent_type}[{wr.worker_idx}]: {wr.error}")
                continue

            r = wr.result

            if wr.agent_type == "generation":
                for h in r.get("hypotheses", []):
                    if isinstance(h, dict):
                        continue  # skip already-serialized dicts; pool expects Hypothesis objects
                    pool.add(h)
                    new_hypotheses.append(h)

            elif wr.agent_type == "reflection":
                for h in r.get("hypotheses", []):
                    existing = pool.get(h.id) if hasattr(h, "id") else None
                    if existing:
                        for c in getattr(h, "criticisms", []):
                            existing.add_criticism(c)
                            total_criticisms += 1

            elif wr.agent_type == "evolution":
                for h in r.get("evolved_hypotheses", []):
                    if isinstance(h, dict):
                        continue
                    pool.add(h)
                    new_hypotheses.append(h)

            elif wr.agent_type == "search":
                hyp_id: str = r.get("hypothesis_id", "")
                synthesis: str = r.get("synthesis", "")
                if hyp_id and synthesis:
                    existing = pool.get(hyp_id)
                    if existing:
                        existing.add_evidence(Evidence(
                            source="swarm_search",
                            evidence_type="web_research",
                            details=synthesis,
                            confidence=0.75,
                        ))
                        total_evidence += 1

        return {
            "new_hypotheses": new_hypotheses,
            "total_criticisms": total_criticisms,
            "total_evidence": total_evidence,
            "errors": errors,
        }
