"""IntentParser — classify a natural-language task and extract constraints
(CLAUDE.md §3, Phase 1 routing).

Routes scientific hypothesis-generation requests to the preserved Co-Scientist
research pipeline; everything else goes to the JARVIS worker loop. Heuristic-first
(simplest that works, §11.2), with an optional LLM tie-breaker.
"""

from __future__ import annotations

import re
from typing import Dict, Optional

from pydantic import BaseModel

_RESEARCH_PATTERNS = [
    r"\bhypothes",  # hypothesis / hypotheses / hypothesize
    r"\bnovel\s+.*\b(target|mechanism|therap|treatment|drug|intervention)",
    r"\bresearch\s+(question|goal|directions?)\b",
    r"\bgenerate\s+(hypotheses|research)\b",
    r"\b(drug|therapeutic)\s+targets?\b",
    r"\bmechanism of action\b",
    r"\bdiscover\b.*\b(targets?|candidates?|mechanisms?)\b",
]


class Intent(BaseModel):
    kind: str  # "research" | "task"
    goal: str
    constraints: Dict[str, str] = {}
    reason: str = ""


class _LLMIntent(BaseModel):
    kind: str
    reason: str


class IntentParser:
    def __init__(self, llm_router=None):
        self.llm = llm_router

    def _heuristic(self, text: str) -> Optional[str]:
        low = text.lower()
        for pat in _RESEARCH_PATTERNS:
            if re.search(pat, low):
                return "research"
        return None

    async def parse(self, text: str) -> Intent:
        text = text.strip()
        constraints = _extract_constraints(text)

        hinted = self._heuristic(text)
        if hinted == "research":
            return Intent(
                kind="research",
                goal=text,
                constraints=constraints,
                reason="matched research/hypothesis pattern",
            )

        # Default to the general worker. Optionally let the LLM upgrade to research.
        if self.llm is not None:
            try:
                verdict = await self.llm.complete(
                    [
                        {
                            "role": "system",
                            "content": "Classify the task as 'research' (open-ended "
                            "scientific hypothesis generation / discovery) or 'task' (any concrete, "
                            'executable job). Reply JSON {"kind": "research"|"task", "reason": str}.',
                        },
                        {"role": "user", "content": text},
                    ],
                    response_format=_LLMIntent,
                )
                if isinstance(verdict, _LLMIntent) and verdict.kind in (
                    "research",
                    "task",
                ):
                    return Intent(
                        kind=verdict.kind,
                        goal=text,
                        constraints=constraints,
                        reason=verdict.reason,
                    )
            except Exception:
                pass

        return Intent(
            kind="task",
            goal=text,
            constraints=constraints,
            reason="default → worker loop",
        )


def _extract_constraints(text: str) -> Dict[str, str]:
    """Pull lightweight constraints (output path, deadlines) from the request."""
    constraints: Dict[str, str] = {}
    out = re.search(r"(?:->|→|to)\s*(\.?/?[\w./\-]+\.\w{2,4})\b", text)
    if out:
        constraints["output_path"] = out.group(1)
    return constraints
