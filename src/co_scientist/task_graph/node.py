"""Task graph node + status (CLAUDE.md §1.2 Phase 3).

A ``TaskNode`` is one unit of decomposed work — a goal plus its dependency edges
into the DAG. The scheduler walks the graph, dispatching READY nodes (all deps
DONE) to Workers and recording the outcome back onto the node.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class NodeStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


def _short_id() -> str:
    return uuid.uuid4().hex[:8]


class TaskNode(BaseModel):
    """One node in the hierarchical task DAG (fields per CLAUDE.md §1.2)."""

    id: str = Field(default_factory=_short_id)
    goal: str
    dependencies: List[str] = []
    status: NodeStatus = NodeStatus.PENDING
    artifacts: List[str] = []
    verification: Optional[str] = None
    assigned_to: Optional[str] = None
    retry_count: int = 0
    error_history: List[str] = []
    result: Optional[str] = None
    depth: int = 0
