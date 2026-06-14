"""Ready-node selection helper (CLAUDE.md §3 file layout).

Thin facade over the DAG: the actual concurrent execution lives in
``co_scientist.task_graph.scheduler.DagScheduler``, which is re-exported here for
convenience.
"""

from __future__ import annotations

from co_scientist.task_graph.scheduler import DagScheduler

__all__ = ["DagScheduler"]
