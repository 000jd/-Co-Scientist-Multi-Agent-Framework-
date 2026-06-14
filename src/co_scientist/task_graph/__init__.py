"""Hierarchical task graph (CLAUDE.md §1.2, §3, §7 Phase 3).

Decompose a goal into a DAG of subtasks, schedule ready nodes to Workers in
parallel, and isolate failures so an unrelated branch cannot sink the whole run.
"""

from co_scientist.task_graph.graph import TaskGraph
from co_scientist.task_graph.node import NodeStatus, TaskNode

__all__ = ["TaskNode", "NodeStatus", "TaskGraph"]
