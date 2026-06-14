"""AuditLogger — file-level append-only audit trail (CLAUDE.md §6 Phase 6).

Every read / write / command is recorded as one JSON line with a timestamp, a
sha256 digest of the payload, and a redacted/truncated preview. The log is the
source of truth for replay and rollback after a crash.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import structlog
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = structlog.get_logger("audit_logger")

_SENSITIVE_KEYS = {"api_key", "token", "password", "secret", "authorization"}
# Actions that mutate state — these are what a rollback would have to undo.
_MUTATING = {"file_write", "write_file", "edit_file", "file_edit", "command"}

# Secret-looking substrings to scrub from VALUES (command/content leak secrets even
# when the key name is innocuous). Each pattern is replaced with [REDACTED].
_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{12,}"),  # OpenAI/Anthropic-style keys
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{8,}"),  # Authorization: Bearer <token>
    re.compile(r"AKIA[0-9A-Z]{12,}"),  # AWS access key id
    re.compile(r"\b[A-Za-z0-9_\-]{40,}\b"),  # long high-entropy tokens
]


def _scrub_secrets(value: str) -> str:
    for pat in _SECRET_PATTERNS:
        value = pat.sub("[REDACTED]", value)
    return value


class AuditLogger:
    """Append-only JSON-lines audit log; one line per action. Writes are serialized
    with an asyncio.Lock so concurrent agents never interleave lines."""

    def __init__(self, path: str = "./data/audit/audit.jsonl"):
        self.log_path = Path(path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def record(
        self,
        action_type: str,
        payload: Dict[str, Any],
        agent: Optional[str] = None,
        cycle: Optional[int] = None,
    ) -> None:
        """Append one audit record (timestamp, digest, redacted preview)."""
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "action": action_type,
            "agent": agent,
            "cycle": cycle,
            "payload_digest": hashlib.sha256(
                json.dumps(payload, default=str, sort_keys=True).encode()
            ).hexdigest()[:12],
            "payload_preview": self._redact(payload),
        }
        async with self._lock:
            try:
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
            except Exception as e:  # noqa: BLE001
                logger.warning("audit_write_failed", error=str(e))

    @staticmethod
    def _redact(payload: Dict[str, Any]) -> Dict[str, str]:
        redacted: Dict[str, str] = {}
        for k, v in payload.items():
            if any(s in k.lower() for s in _SENSITIVE_KEYS):
                redacted[k] = "[REDACTED]"
            else:
                # Scrub secret-looking VALUES too, then truncate.
                redacted[k] = _scrub_secrets(str(v))[:200]
        return redacted

    async def _read_all(self) -> List[Dict[str, Any]]:
        async with self._lock:
            try:
                text = self.log_path.read_text()
            except Exception:  # noqa: BLE001
                return []
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    async def get_recent(self, n: int = 50) -> List[Dict[str, Any]]:
        """Return the last N audit records (ordered oldest→newest)."""
        records = await self._read_all()
        return records[-n:]

    async def replay(self) -> List[Dict[str, Any]]:
        """Return the full ordered log — a crashed task is reconstructed from this."""
        return await self._read_all()

    async def rollback(self, to_index: int) -> List[Dict[str, Any]]:
        """Return the mutating actions recorded AFTER ``to_index`` — i.e. the tail a
        caller would reverse to undo back to that checkpoint. ``to_index`` is the index
        of the last record to KEEP (use -1 to mean 'undo everything').

        Each returned entry carries the original action plus the ``target`` (file_write
        path / command) so the caller can reconstruct what was touched.
        """
        records = await self._read_all()
        tail = records[to_index + 1 :]
        undo: List[Dict[str, Any]] = []
        for rec in tail:
            if rec.get("action") not in _MUTATING:
                continue
            preview = rec.get("payload_preview", {})
            undo.append(
                {
                    "action": rec.get("action"),
                    "ts": rec.get("ts"),
                    "target": preview.get("path") or preview.get("command"),
                    "preview": preview,
                }
            )
        # Reverse order: undo the most recent mutation first.
        undo.reverse()
        return undo
