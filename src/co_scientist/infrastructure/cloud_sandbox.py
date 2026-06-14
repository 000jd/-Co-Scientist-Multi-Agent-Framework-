"""E2B cloud-sandbox wrapper (CLAUDE.md §3 infrastructure/cloud_sandbox.py, §7 Phase 5).

The E2B execution path lives in :class:`ExecutionSandbox` (provider="e2b"); this module
is the documented seam for E2B-specific configuration and a key/SDK availability probe.
Requires ``E2B_API_KEY`` + the ``e2b`` / ``e2b_code_interpreter`` SDK; without them the
unified sandbox transparently falls back to docker → local.
"""

from __future__ import annotations

import os
from typing import Optional

import structlog

from co_scientist.infrastructure.execution_sandbox import (
    ExecutionSandbox,
    _e2b_available,
)

log = structlog.get_logger("infrastructure.cloud_sandbox")


def e2b_ready(api_key: Optional[str] = None) -> bool:
    """True iff an E2B key + SDK are present (so provider='e2b' will be used, not fall back)."""
    return _e2b_available(api_key or os.environ.get("E2B_API_KEY"))


class CloudSandbox(ExecutionSandbox):
    """ExecutionSandbox pinned to the E2B provider (still falls back if unavailable)."""

    def __init__(self, sandbox_config, workspace_root: str = "./projects"):
        # Force the requested provider to e2b; resolution still degrades gracefully.
        try:
            sandbox_config.provider = "e2b"
        except Exception:  # frozen/!mutable config — resolution handles it anyway
            pass
        super().__init__(sandbox_config, workspace_root=workspace_root)
        if self.provider != "e2b":
            log.info("cloud_sandbox.e2b_unavailable_fell_back", provider=self.provider)
