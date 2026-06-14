"""Decompose a goal into a task DAG (CLAUDE.md §1.2, §3 Phase 3).

Asks the LLM to split a goal into subtasks with explicit dependency edges, then
builds a validated ``TaskGraph``. Falls back to a single-node graph when there is
no LLM or the call fails, so the master agent always has something to run.
"""

from __future__ import annotations

from typing import List, Optional

import structlog
from pydantic import BaseModel

from co_scientist.task_graph.graph import TaskGraph
from co_scientist.task_graph.node import TaskNode

log = structlog.get_logger("agent.task_graph_builder")

_SYSTEM = """You are a planning agent. Decompose the user's GOAL into a small DAG of \
concrete subtasks. Reply with JSON: {"nodes": [{"id": str, "goal": str, \
"dependencies": [ids], "verification": str|null}, ...]}.

Rules:
- Use short stable ids (e.g. "n1", "n2").
- "dependencies" lists ids of tasks that must finish first. Independent tasks have [].
- The graph MUST be acyclic. Keep it minimal — only split when subtasks are \
genuinely independent or sequential.
- "verification" is an optional shell command that exits 0 iff the subtask is done."""


class _PlanNode(BaseModel):
    id: str
    goal: str
    dependencies: List[str] = []
    verification: Optional[str] = None


class _PlanGraph(BaseModel):
    nodes: List[_PlanNode] = []


class TaskGraphBuilder:
    """Builds a ``TaskGraph`` from a natural-language goal."""

    def __init__(self, llm_router=None, max_depth: int = 4):
        self.llm = llm_router
        self.max_depth = max(1, int(max_depth))

    async def build(self, goal: str) -> TaskGraph:
        if self.llm is not None:
            try:
                plan = await self.llm.complete(
                    [
                        {"role": "system", "content": _SYSTEM},
                        {"role": "user", "content": f"GOAL:\n{goal}"},
                    ],
                    response_format=_PlanGraph,
                )
                if isinstance(plan, _PlanGraph) and plan.nodes:
                    return self._from_plan(plan, goal)
            except Exception as e:  # noqa: BLE001
                log.warning("task_graph_builder.llm_failed", error=str(e))

        return self._single_node(goal)

    # ── construction + validation ───────────────────────────────────────────
    def _from_plan(self, plan: _PlanGraph, goal: str) -> TaskGraph:
        graph = TaskGraph()
        ids = {n.id for n in plan.nodes}
        for n in plan.nodes:
            # Drop edges to ids that don't exist in the plan.
            deps = [d for d in n.dependencies if d in ids and d != n.id]
            graph.add_node(
                TaskNode(
                    id=n.id, goal=n.goal, dependencies=deps, verification=n.verification
                )
            )

        self._break_cycles(graph)
        self._assign_depths(graph)
        return graph

    def _break_cycles(self, graph: TaskGraph) -> None:
        """Drop back-edges until the graph is acyclic (DFS, deterministic order)."""
        guard = 0
        while graph.has_cycle() and guard < len(graph.nodes) + 1:
            guard += 1
            if not self._remove_one_back_edge(graph):
                break

    def _remove_one_back_edge(self, graph: TaskGraph) -> bool:
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {nid: WHITE for nid in graph.nodes}

        def visit(nid: str) -> bool:
            color[nid] = GRAY
            node = graph.nodes[nid]
            for dep in list(node.dependencies):
                if dep not in color:
                    continue
                if color[dep] == GRAY:
                    node.dependencies.remove(dep)  # back-edge → break it
                    log.warning(
                        "task_graph_builder.dropped_back_edge", node=nid, dep=dep
                    )
                    return True
                if color[dep] == WHITE and visit(dep):
                    return True
            color[nid] = BLACK
            return False

        return any(color[nid] == WHITE and visit(nid) for nid in graph.nodes)

    def _assign_depths(self, graph: TaskGraph) -> None:
        """Set each node's depth = longest path from a root; cap at ``max_depth``.

        Nodes deeper than the cap have their excess dependencies pruned so the
        DAG stays within ``max_depth`` levels (CLAUDE.md §1.2 depth cap).
        """
        memo: dict[str, int] = {}

        def depth(nid: str, stack: set) -> int:
            if nid in memo:
                return memo[nid]
            if nid in stack:
                return 0  # cycle already broken, but be safe
            stack.add(nid)
            node = graph.nodes[nid]
            deps = [d for d in node.dependencies if d in graph.nodes]
            d = 0 if not deps else 1 + max(depth(p, stack) for p in deps)
            stack.discard(nid)
            memo[nid] = d
            return d

        for nid, node in graph.nodes.items():
            d = depth(nid, set())
            if d >= self.max_depth:
                node.dependencies = []  # collapse to a root to respect the cap
                node.depth = 0
            else:
                node.depth = d

    def _single_node(self, goal: str) -> TaskGraph:
        graph = TaskGraph()
        graph.add_node(TaskNode(goal=goal, dependencies=[], depth=0))
        return graph
