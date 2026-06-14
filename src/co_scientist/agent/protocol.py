"""Worker action protocol (CLAUDE.md §1.1 Plan stage).

The model emits exactly one of::

    tool_call({"tool": "bash", "args": {"command": "..."}})
    FINAL_ANSWER({"answer": "...", "verification": "<optional shell check>"})

``parse_action`` extracts a structured :class:`Action` from raw model text, tolerant
of surrounding prose and ```json fences.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Action:
    kind: str  # "tool_call" | "final_answer" | "noop"
    tool: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    answer: Optional[str] = None
    verification: Optional[str] = (
        None  # shell command; exit 0 == verified (deterministic)
    )
    raw: str = ""


def _extract_call(text: str, keyword: str) -> Optional[str]:
    """Return the JSON payload inside ``keyword(...)`` using balanced-paren scanning."""
    idx = text.rfind(keyword + "(")
    if idx == -1:
        return None
    start = idx + len(keyword) + 1
    depth = 1
    i = start
    in_str = False
    esc = False
    while i < len(text):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    return text[start:i].strip()
        i += 1
    return None


def _loads(payload: str) -> Optional[Dict[str, Any]]:
    payload = payload.strip()
    if payload.startswith("```"):
        payload = payload.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        obj = json.loads(payload)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def parse_action(text: str) -> Action:
    """Parse the worker's Plan-stage output into an :class:`Action`.

    FINAL_ANSWER takes precedence over tool_call when both appear.
    """
    if not text:
        return Action(kind="noop", raw=text)

    final = _extract_call(text, "FINAL_ANSWER")
    if final is not None:
        obj = _loads(final) or {}
        return Action(
            kind="final_answer",
            answer=str(obj.get("answer", obj.get("result", ""))),
            verification=obj.get("verification"),
            args=obj,
            raw=text,
        )

    call = _extract_call(text, "tool_call")
    if call is not None:
        obj = _loads(call) or {}
        return Action(
            kind="tool_call",
            tool=obj.get("tool"),
            args=obj.get("args", {}) if isinstance(obj.get("args"), dict) else {},
            raw=text,
        )

    return Action(kind="noop", raw=text)
