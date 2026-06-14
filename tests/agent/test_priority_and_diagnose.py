"""PriorityScheduler ordering + DiagnoseAgent post-mortem (CLAUDE.md §1.4/§3)."""

import pytest

from co_scientist.agent.priority_scheduler import PriorityScheduler
from co_scientist.agent.diagnoser import DiagnoseAgent
from co_scientist.task_graph.graph import TaskGraph
from co_scientist.task_graph.node import TaskNode


def test_priority_orders_high_impact_first():
    g = TaskGraph()
    g.add_node(TaskNode(id="root", goal="root", dependencies=[]))
    g.add_node(TaskNode(id="leaf", goal="leaf", dependencies=[]))
    g.add_node(TaskNode(id="c1", goal="c1", dependencies=["root"]))
    g.add_node(TaskNode(id="c2", goal="c2", dependencies=["root"]))
    ordered = PriorityScheduler.order(g, [g.get("leaf"), g.get("root")])
    # root unblocks 2 dependents → must come before the leaf
    assert [n.id for n in ordered] == ["root", "leaf"]


@pytest.mark.asyncio
async def test_diagnose_heuristic_explains_max_turns():
    d = await DiagnoseAgent(llm_router=None).diagnose("do a thing", "max_turns", [])
    assert "turn cap" in d.root_cause.lower()
    assert d.suggestion


@pytest.mark.asyncio
async def test_diagnose_heuristic_explains_budget():
    d = await DiagnoseAgent(llm_router=None).diagnose("do a thing", "budget", [])
    assert "budget" in d.root_cause.lower()
