"""
AuditLogger — records every agent action to a persistent audit trail.

Kimi Work security requirement: every action (read, write, browse, LLM call)
is logged with timestamp, agent, action type, and payload digest.
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("audit_logger")


class AuditLogger:
    """Append-only JSON-lines audit log. One line per action."""
    
    def __init__(self, log_path: str = "./data/audit/audit.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        
    async def record(
        self,
        action_type: str,
        payload: Dict[str, Any],
        agent: Optional[str] = None,
        cycle: Optional[int] = None,
    ) -> None:
        """
        Append one audit record. Non-blocking — fire and forget.
        action_type examples: swarm_plan, task_dispatch, task_complete,
                              llm_call, browse, file_read, file_write,
                              governance_blocked, governance_approved
        """
        SENSITIVE_KEYS = {"api_key", "token", "password", "secret", "authorization"}
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "action": action_type,
            "agent": agent,
            "cycle": cycle,
            "payload_digest": hashlib.sha256(
                json.dumps(payload, default=str, sort_keys=True).encode()
            ).hexdigest()[:12],
            "payload_preview": {
                k: "[REDACTED]" if any(s in k.lower() for s in SENSITIVE_KEYS) else str(v)[:200]
                for k, v in payload.items()
            },
        }
        
        async with self._lock:
            try:
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
            except Exception as e:
                logger.warning(f"AuditLogger write failed: {e}")
                
    async def get_recent(self, n: int = 50) -> list:
        """Return last N audit records."""
        try:
            lines = self.log_path.read_text().strip().split("\n")
            return [json.loads(l) for l in lines[-n:] if l.strip()]
        except Exception:
            return []
