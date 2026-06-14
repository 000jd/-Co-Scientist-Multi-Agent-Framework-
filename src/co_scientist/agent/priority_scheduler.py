"""PriorityScheduler — order ready DAG nodes by impact (CLAUDE.md §1.4/§3, was RankingAgent).

Ready nodes are dispatched highest-impact first: a node that unblocks more downstream
work (more dependents) and sits earlier in the graph runs before a leaf. Pure, deterministic.
"""

from __future__ import annotations

from typing import List

from co_scientist.task_graph.graph import TaskGraph
from co_scientist.task_graph.node import TaskNode


class PriorityScheduler:
    @staticmethod
    def order(graph: TaskGraph, nodes: List[TaskNode]) -> List[TaskNode]:
        """Return ``nodes`` sorted by descending priority (dependents desc, depth asc, id)."""
        dependents = _dependent_counts(graph)
        return sorted(
            nodes,
            key=lambda n: (-dependents.get(n.id, 0), n.depth, n.id),
        )


def _dependent_counts(graph: TaskGraph) -> dict:
    counts: dict = {}
    for node in graph.all_nodes():
        for dep in node.dependencies:
            counts[dep] = counts.get(dep, 0) + 1
    return counts
