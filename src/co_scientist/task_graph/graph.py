"""The task DAG container (CLAUDE.md §1.2, §3 Phase 3).

Holds the nodes and answers the questions the scheduler asks: which nodes are
ready, is the run complete, does the graph have a cycle, and which nodes are now
unreachable because a dependency failed.
"""

from __future__ import annotations

from typing import Dict, List

from co_scientist.task_graph.node import NodeStatus, TaskNode


class TaskGraph:
    """A directed acyclic graph of ``TaskNode``s keyed by node id."""

    def __init__(self) -> None:
        self.nodes: Dict[str, TaskNode] = {}

    def add_node(self, node: TaskNode) -> TaskNode:
        self.nodes[node.id] = node
        return node

    def get(self, node_id: str) -> TaskNode | None:
        return self.nodes.get(node_id)

    def all_nodes(self) -> List[TaskNode]:
        return list(self.nodes.values())

    def ready_nodes(self) -> List[TaskNode]:
        """Nodes that can run now: PENDING/READY with every dependency DONE.

        A missing dependency id is treated as unsatisfiable (the node is not
        ready); it is only resolved to FAILED via ``unreachable_nodes``.
        """
        ready: List[TaskNode] = []
        for node in self.nodes.values():
            if node.status not in (NodeStatus.PENDING, NodeStatus.READY):
                continue
            deps = [self.nodes.get(d) for d in node.dependencies]
            if all(dep is not None and dep.status == NodeStatus.DONE for dep in deps):
                ready.append(node)
        return ready

    def mark(self, node_id: str, status: NodeStatus) -> None:
        node = self.nodes.get(node_id)
        if node is not None:
            node.status = status

    def is_complete(self) -> bool:
        """True once every node has settled (DONE or FAILED)."""
        return all(
            n.status in (NodeStatus.DONE, NodeStatus.FAILED)
            for n in self.nodes.values()
        )

    def has_cycle(self) -> bool:
        """Detect a dependency cycle via DFS with a recursion stack."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {nid: WHITE for nid in self.nodes}

        def visit(nid: str) -> bool:
            color[nid] = GRAY
            for dep in self.nodes[nid].dependencies:
                if dep not in color:
                    continue  # dangling edge — not part of any cycle
                if color[dep] == GRAY:
                    return True
                if color[dep] == WHITE and visit(dep):
                    return True
            color[nid] = BLACK
            return False

        return any(color[nid] == WHITE and visit(nid) for nid in self.nodes)

    def unreachable_nodes(self) -> List[TaskNode]:
        """Nodes that can never run because some ancestor FAILED (or is missing).

        Only considers nodes still in flight (PENDING/READY/RUNNING); already
        settled nodes are excluded.
        """
        unreachable: List[TaskNode] = []
        for node in self.nodes.values():
            if node.status in (NodeStatus.DONE, NodeStatus.FAILED):
                continue
            if self._has_blocked_ancestor(node.id, set()):
                unreachable.append(node)
        return unreachable

    def _has_blocked_ancestor(self, node_id: str, seen: set) -> bool:
        if node_id in seen:
            return False
        seen.add(node_id)
        node = self.nodes.get(node_id)
        if node is None:
            return False
        for dep_id in node.dependencies:
            dep = self.nodes.get(dep_id)
            if dep is None or dep.status == NodeStatus.FAILED:
                return True
            if self._has_blocked_ancestor(dep_id, seen):
                return True
        return False
