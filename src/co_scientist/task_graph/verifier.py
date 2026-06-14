"""Per-node verification for the task graph (CLAUDE.md §1.2, mirrors agent/verifier.py).

NOTE: in the live path, a node's ``verification`` runs inside the assigned Worker's
Verify stage (``CommandCenter`` passes ``node.verification`` to ``Worker.run``), where it
is Guardian-gated and project-scoped. This standalone helper is for direct/offline node
checks and is likewise gated + workdir-scoped (review #20).

A node passes if its ``verification`` shell command exits 0 (deterministic, preferred),
otherwise if it produced a non-empty ``result``.
"""

from __future__ import annotations

from typing import Optional

import structlog

from co_scientist.task_graph.node import TaskNode

log = structlog.get_logger("task_graph.verifier")


class NodeVerifier:
    """Verifies a completed ``TaskNode`` (deterministic command, else heuristic)."""

    def __init__(self, sandbox=None, guardian=None, workdir: Optional[str] = None):
        self.sandbox = sandbox
        self.guardian = guardian
        self.workdir = workdir

    async def verify(self, node: TaskNode) -> bool:
        # 1. Deterministic shell check (preferred — reproducible, no LLM).
        if node.verification and self.sandbox is not None:
            if (
                self.guardian is not None
            ):  # gate the model-supplied command (review #20)
                verdict = await self.guardian.check_action(
                    "command", node.verification, {"node": node.id}
                )
                if not verdict.approved:
                    log.warning(
                        "node_verifier.blocked", node=node.id, reason=verdict.reason
                    )
                    return False
            result = await self.sandbox.run(node.verification, workdir=self.workdir)
            log.info("node_verifier.command", node=node.id, exit_code=result.exit_code)
            return bool(result.success)

        # 2. Conservative heuristic: a non-empty result.
        return bool(node.result and node.result.strip())
