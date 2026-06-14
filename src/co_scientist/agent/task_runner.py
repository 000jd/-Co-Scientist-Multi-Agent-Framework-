"""run_task — the single entry behind ``co-scientist task "..."`` (CLAUDE.md §7 Phase 1/2).

Parses intent, then either routes to the preserved Co-Scientist research pipeline
or drives the JARVIS 5-stage worker loop. The existing ``research`` command path is
untouched.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import structlog

from co_scientist.agent.intent_parser import IntentParser
from co_scientist.agent.worker import Worker
from co_scientist.core.config import Config
from co_scientist.infrastructure.execution_sandbox import make_sandbox
from co_scientist.infrastructure.llm import LLMRouter

log = structlog.get_logger("agent.task_runner")


def _llm_configured(config: Config) -> bool:
    chain = config.llm.fallback_chain or []
    if not chain:
        return False
    # ollama needs no key; cloud providers need one
    if any(p in chain for p in ("ollama",)):
        return True
    return config.llm.api_key is not None


async def run_task(
    goal: str,
    config: Config,
    *,
    project_id: Optional[str] = None,
    workspace_root: str = "./projects",
    force_kind: Optional[str] = None,
    use_graph: bool = False,
) -> Dict[str, Any]:
    """Execute an arbitrary natural-language task and return a structured result."""
    if not _llm_configured(config):
        return {
            "kind": "error",
            "success": False,
            "answer": "",
            "message": "No LLM provider configured. Set llm.fallback_chain + an API key "
            "(e.g. LLM_MODEL, LLM_BASE_URL, OPENROUTER_API_KEY) or run a local Ollama.",
        }

    llm = LLMRouter(config.llm)
    intent_kind = force_kind
    if intent_kind is None:
        intent = await IntentParser(llm_router=llm).parse(goal)
        intent_kind = intent.kind

    if intent_kind == "research":
        return await _run_research(goal, config)

    # ── Hierarchical decomposition path (Phase 3): CommandCenter → DAG → workers ──
    if use_graph:
        from co_scientist.agent.command_center import CommandCenter

        sandbox = make_sandbox(config, workspace_root=workspace_root)
        cc = CommandCenter(config, llm, sandbox)
        result = await cc.execute(goal)
        result["kind"] = "task"
        result["cost_usd"] = round(getattr(llm.cost_tracker, "current_total", 0.0), 4)
        result["success"] = len(result.get("failed", [])) == 0 and bool(
            result.get("done")
        )
        return result

    # ── JARVIS worker loop ────────────────────────────────────────────────
    project_id = project_id or f"task-{uuid.uuid4().hex[:8]}"
    workdir = Path(workspace_root) / project_id
    sandbox = make_sandbox(config, workspace_root=workspace_root)

    guardian = audit = None
    try:  # governance is optional; wire it when present (Phase 6)
        from co_scientist.governance.guardian_agent import GuardianAgent
        from co_scientist.governance.audit_logger import AuditLogger

        audit = AuditLogger(config.agents.audit_log_path)
        guardian = GuardianAgent(audit_logger=audit)
    except Exception:
        pass

    worker = Worker(
        llm_router=llm,
        sandbox=sandbox,
        guardian=guardian,
        audit=audit,
        max_turns=config.agent.max_turns,
        budget_usd=config.agent.budget_usd,
        project_id=project_id,
        workdir=str(workdir),
    )
    result = await worker.run(goal)
    diagnosis = None
    if not result.success:
        from co_scientist.agent.diagnoser import DiagnoseAgent

        d = await DiagnoseAgent(llm_router=llm).diagnose(
            goal, result.stop_reason, result.transcript
        )
        diagnosis = f"{d.root_cause} — {d.suggestion}"
    return {
        "kind": "task",
        "success": result.success,
        "answer": result.answer,
        "diagnosis": diagnosis,
        "verified": result.verified,
        "verification_reason": result.verification_reason,
        "turns": result.turns,
        "stop_reason": result.stop_reason,
        "artifacts": result.artifacts,
        "project_id": project_id,
        "workdir": str(workdir),
        "cost_usd": round(getattr(llm.cost_tracker, "current_total", 0.0), 4),
    }


async def _run_research(goal: str, config: Config) -> Dict[str, Any]:
    """Route to the preserved hypothesis pipeline (called as a sub-tool)."""
    from co_scientist.core.orchestrator import CoScientistOrchestrator

    orchestrator = CoScientistOrchestrator(config, config.get_candidate_model_class())
    await orchestrator.run_pipeline(goal, max_cycles=config.experiment.max_cycles)
    leaderboard = orchestrator.leaderboard.to_display()
    return {
        "kind": "research",
        "success": True,
        "answer": "research pipeline complete",
        "leaderboard": leaderboard[:10],
    }
