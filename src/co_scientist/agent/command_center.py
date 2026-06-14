"""CommandCenter — the master agent that runs a goal as a task DAG (CLAUDE.md §1.4).

(Was SupervisorAgent.) Builds the ``TaskGraph`` with ``TaskGraphBuilder``, then
drives ``DagScheduler`` with a worker_factory that spins up one ``Worker`` per
node. Independent branches run in parallel; a failed node only sinks its own
descendants.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from co_scientist.agent.task_graph_builder import TaskGraphBuilder
from co_scientist.agent.worker import Worker
from co_scientist.task_graph.node import TaskNode
from co_scientist.task_graph.scheduler import DagScheduler

log = structlog.get_logger("agent.command_center")


class CommandCenter:
    """Top-level orchestrator: goal → DAG → parallel Workers → aggregated result."""

    def __init__(self, config, llm_router, sandbox, guardian=None, audit=None):
        self.config = config
        self.llm = llm_router
        self.sandbox = sandbox
        self.builder = TaskGraphBuilder(
            llm_router=llm_router, max_depth=config.agent.max_task_depth
        )
        # Every DAG worker must be gated + audited like the single-worker path
        # (review #6/#12). Build the governance stack if the caller didn't supply one.
        if guardian is None or audit is None:
            try:
                from co_scientist.governance.audit_logger import AuditLogger
                from co_scientist.governance.guardian_agent import GuardianAgent

                audit = audit or AuditLogger(config.agents.audit_log_path)
                guardian = guardian or GuardianAgent(audit_logger=audit)
            except Exception:  # governance optional
                pass
        self.guardian = guardian
        self.audit = audit

    async def execute(self, goal: str) -> dict:
        graph = await self.builder.build(goal)
        base = _slug(goal)
        workdir_root = getattr(self.sandbox, "workspace_root", Path("./projects"))

        def worker_factory(node: TaskNode):
            project_id = f"{base}-{node.id}"
            worker = Worker(
                llm_router=self.llm,
                sandbox=self.sandbox,
                guardian=self.guardian,
                audit=self.audit,
                max_turns=self.config.agent.max_turns,
                budget_usd=self.config.agent.budget_usd,
                project_id=project_id,
                workdir=str(Path(workdir_root) / project_id),
            )
            node.assigned_to = project_id
            # The planner's deterministic node check runs in the worker's Verify stage (review #1).
            return worker.run(node.goal, verification=node.verification)

        scheduler = DagScheduler(max_workers=self.config.swarm.max_workers)
        summary = await scheduler.run(graph, worker_factory)

        artifacts: list[str] = []
        for n in graph.all_nodes():
            artifacts.extend(n.artifacts)

        return {
            "goal": goal,
            "nodes": [n.model_dump(mode="json") for n in graph.all_nodes()],
            "artifacts": artifacts,
            "done": summary["done"],
            "failed": summary["failed"],
        }


def _slug(goal: str) -> str:
    safe = "".join(c if c.isalnum() else "-" for c in goal.lower()).strip("-")
    return (safe[:32] or "task").rstrip("-")
