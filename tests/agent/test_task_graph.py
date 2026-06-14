"""Phase 3 tests — hierarchical task graph (CLAUDE.md §7).

Drive the REAL TaskGraphBuilder / DagScheduler / CommandCenter with a scripted
mock LLM and a local ExecutionSandbox. Fully deterministic, no network / API key.
"""

import asyncio
import json
from pathlib import Path

import pytest

from co_scientist.agent.command_center import CommandCenter
from co_scientist.agent.task_graph_builder import TaskGraphBuilder
from co_scientist.core.config import Config, SandboxConfig
from co_scientist.infrastructure.execution_sandbox import ExecutionSandbox
from co_scientist.task_graph.graph import TaskGraph
from co_scientist.task_graph.node import NodeStatus, TaskNode
from co_scientist.task_graph.scheduler import DagScheduler
from tests.mock_llm import MockLLMRouter


def _local_sandbox(tmp_path: Path) -> ExecutionSandbox:
    cfg = SandboxConfig(provider="local", local_timeout_seconds=30)
    return ExecutionSandbox(cfg, workspace_root=str(tmp_path))


class _FakeResult:
    def __init__(self, success=True, answer="ok", artifacts=None):
        self.success = success
        self.answer = answer
        self.artifacts = artifacts or []
        self.verification_reason = "" if success else "did not verify"
        self.stop_reason = "final_answer" if success else "no_progress"


# ── builder: scripted DAG JSON → 3 nodes with correct deps, no cycle ─────────
async def test_builder_produces_dag_with_deps():
    plan = {
        "nodes": [
            {"id": "n1", "goal": "gather data", "dependencies": []},
            {"id": "n2", "goal": "analyze data", "dependencies": ["n1"]},
            {"id": "n3", "goal": "write report", "dependencies": ["n2"]},
        ]
    }
    llm = MockLLMRouter(responses=[json.dumps(plan)])
    builder = TaskGraphBuilder(llm_router=llm, max_depth=4)

    graph = await builder.build("produce a research report")

    assert len(graph.all_nodes()) == 3
    assert graph.get("n2").dependencies == ["n1"]
    assert graph.get("n3").dependencies == ["n2"]
    assert graph.has_cycle() is False
    assert graph.get("n1").depth == 0
    assert graph.get("n3").depth == 2


async def test_builder_falls_back_to_single_node_without_llm():
    builder = TaskGraphBuilder(llm_router=None)
    graph = await builder.build("just do the thing")
    nodes = graph.all_nodes()
    assert len(nodes) == 1
    assert nodes[0].goal == "just do the thing"
    assert nodes[0].dependencies == []


async def test_builder_drops_back_edges_to_stay_acyclic():
    plan = {
        "nodes": [
            {"id": "a", "goal": "a", "dependencies": ["b"]},
            {"id": "b", "goal": "b", "dependencies": ["a"]},  # cycle a<->b
        ]
    }
    llm = MockLLMRouter(responses=[json.dumps(plan)])
    graph = await TaskGraphBuilder(llm_router=llm).build("cyclic goal")
    assert graph.has_cycle() is False


# ── scheduler: diamond DAG runs to completion, deps respected ────────────────
async def test_scheduler_runs_diamond_dag_respecting_deps():
    #      n1
    #     /  \
    #   n2    n3
    #     \  /
    #      n4
    graph = TaskGraph()
    graph.add_node(TaskNode(id="n1", goal="root", dependencies=[]))
    graph.add_node(TaskNode(id="n2", goal="left", dependencies=["n1"]))
    graph.add_node(TaskNode(id="n3", goal="right", dependencies=["n1"]))
    graph.add_node(TaskNode(id="n4", goal="join", dependencies=["n2", "n3"]))

    order: list[str] = []
    lock = asyncio.Lock()

    async def factory(node):
        async with lock:
            order.append(node.id)
        await asyncio.sleep(0)  # let other ready nodes interleave
        return _FakeResult(
            success=True, answer=f"done {node.id}", artifacts=[f"{node.id}.txt"]
        )

    summary = await DagScheduler(max_workers=4).run(graph, factory)

    assert summary["failed"] == []
    assert set(summary["done"]) == {"n1", "n2", "n3", "n4"}
    assert all(graph.get(i).status == NodeStatus.DONE for i in ("n1", "n2", "n3", "n4"))
    # ordering must respect dependencies
    assert order.index("n1") < order.index("n2")
    assert order.index("n1") < order.index("n3")
    assert order.index("n2") < order.index("n4")
    assert order.index("n3") < order.index("n4")


# ── fault isolation: one branch fails, the independent branch still completes ─
async def test_scheduler_isolates_failures():
    #   n1 (FAILS) -> n2          n3 -> n4   (independent branch)
    graph = TaskGraph()
    graph.add_node(TaskNode(id="n1", goal="will fail", dependencies=[]))
    graph.add_node(TaskNode(id="n2", goal="depends on failure", dependencies=["n1"]))
    graph.add_node(TaskNode(id="n3", goal="independent root", dependencies=[]))
    graph.add_node(TaskNode(id="n4", goal="independent child", dependencies=["n3"]))

    ran: list[str] = []

    async def factory(node):
        ran.append(node.id)
        await asyncio.sleep(0)
        return _FakeResult(success=(node.id != "n1"))

    summary = await DagScheduler(max_workers=4).run(graph, factory)

    assert graph.get("n1").status == NodeStatus.FAILED
    assert graph.get("n2").status == NodeStatus.FAILED  # unreachable via failed n1
    assert graph.get("n3").status == NodeStatus.DONE  # independent branch survives
    assert graph.get("n4").status == NodeStatus.DONE
    assert set(summary["done"]) == {"n3", "n4"}
    assert set(summary["failed"]) == {"n1", "n2"}
    # n2 must NOT have been dispatched (its only dep failed)
    assert "n2" not in ran
    # the independent branch DID run
    assert "n3" in ran and "n4" in ran


# ── CommandCenter end-to-end on a simple goal with a real local sandbox ──────
async def test_command_center_executes_simple_goal(tmp_path):
    # Single-node plan; the per-node Worker does one bash call then FINAL_ANSWER.
    plan = {"nodes": [{"id": "only", "goal": "echo hello", "dependencies": []}]}

    plans = iter(
        [
            'tool_call({"tool": "bash", "args": {"command": "echo HELLO"}})',
            'FINAL_ANSWER({"answer": "done", "verification": "true"})',
        ]
    )

    def route(messages, rf):
        # The builder asks for a structured _PlanGraph; the worker's verify/repair
        # sub-agents also use response_format. Route by the requested model.
        if rf is not None:
            if "nodes" in rf.model_fields:
                return plan
            return {
                n: (True if f.annotation is bool else "ok")
                for n, f in rf.model_fields.items()
            }
        return next(plans)

    llm = MockLLMRouter(router=route)
    cfg = Config()
    cc = CommandCenter(config=cfg, llm_router=llm, sandbox=_local_sandbox(tmp_path))

    result = await cc.execute("echo hello to a file")

    assert result["goal"] == "echo hello to a file"
    assert result["failed"] == []
    assert result["done"] == ["only"]
    assert len(result["nodes"]) == 1
    assert result["nodes"][0]["status"] == "done"
