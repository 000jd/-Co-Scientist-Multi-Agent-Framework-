"""DAG scheduler — fan ready nodes out to Workers in parallel (CLAUDE.md §1.2, §3).

Dispatches every READY node concurrently (capped by ``max_workers`` via a
semaphore), and as each finishes marks it DONE or FAILED. A FAILED node cascades
ONLY to descendants that become unreachable — independent branches keep running
(fault isolation). Loops until the whole graph has settled.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Dict, List

import structlog

from co_scientist.task_graph.graph import TaskGraph
from co_scientist.task_graph.node import NodeStatus, TaskNode

log = structlog.get_logger("task_graph.scheduler")

WorkerFactory = Callable[[TaskNode], Awaitable]


class DagScheduler:
    """Runs a ``TaskGraph`` to completion with bounded parallelism."""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max(1, int(max_workers))
        self._sem = asyncio.Semaphore(self.max_workers)

    async def run(
        self, graph: TaskGraph, worker_factory: WorkerFactory
    ) -> Dict[str, List[str]]:
        """Schedule ``graph`` until complete; return ``{"done": [...], "failed": [...]}``.

        ``worker_factory(node)`` returns an awaitable yielding a result object with
        ``.success`` / ``.answer`` / ``.artifacts`` (e.g. a ``WorkerResult``).
        """
        running: Dict[asyncio.Task, str] = {}

        from co_scientist.agent.priority_scheduler import PriorityScheduler

        while not graph.is_complete():
            # Dispatch ready nodes highest-impact first, up to the worker cap.
            for node in PriorityScheduler.order(graph, graph.ready_nodes()):
                if node.id in running.values():
                    continue
                if len(running) >= self.max_workers:
                    break
                graph.mark(node.id, NodeStatus.RUNNING)
                task = asyncio.ensure_future(self._run_node(node, worker_factory))
                running[task] = node.id

            if not running:
                # Nothing running and nothing ready → remaining nodes are blocked
                # by a failed/missing ancestor. Settle them and finish.
                for stuck in graph.unreachable_nodes():
                    stuck.error_history.append("unreachable: a dependency failed")
                    graph.mark(stuck.id, NodeStatus.FAILED)
                if graph.is_complete():
                    break
                # Defensive: avoid a hot spin if the graph is somehow wedged.
                log.warning("scheduler.stalled", nodes=len(graph.nodes))
                break

            done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                node_id = running.pop(task)
                self._settle(graph, node_id, task)
                # A failure may have orphaned descendants — fail those, keep the rest.
                for stuck in graph.unreachable_nodes():
                    stuck.error_history.append("unreachable: a dependency failed")
                    graph.mark(stuck.id, NodeStatus.FAILED)

        done_ids = [n.id for n in graph.all_nodes() if n.status == NodeStatus.DONE]
        failed_ids = [n.id for n in graph.all_nodes() if n.status == NodeStatus.FAILED]
        log.info("scheduler.complete", done=len(done_ids), failed=len(failed_ids))
        return {"done": done_ids, "failed": failed_ids}

    async def _run_node(self, node: TaskNode, worker_factory: WorkerFactory):
        async with self._sem:
            return await worker_factory(node)

    def _settle(self, graph: TaskGraph, node_id: str, task: asyncio.Task) -> None:
        node = graph.get(node_id)
        if node is None:
            return
        exc = task.exception()
        if exc is not None:
            node.error_history.append(f"worker raised: {exc}")
            graph.mark(node_id, NodeStatus.FAILED)
            log.warning("scheduler.node_error", node=node_id, error=str(exc))
            return
        result = task.result()
        success = bool(getattr(result, "success", False))
        node.result = getattr(result, "answer", None)
        node.artifacts = list(getattr(result, "artifacts", []) or [])
        if success:
            graph.mark(node_id, NodeStatus.DONE)
        else:
            reason = getattr(result, "verification_reason", "") or getattr(
                result, "stop_reason", ""
            )
            node.error_history.append(f"worker failed: {reason}")
            graph.mark(node_id, NodeStatus.FAILED)
