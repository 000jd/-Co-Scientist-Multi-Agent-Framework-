"""PromptInjectionGuard — sanitizes web/document content before LLM ingestion.

Copied into governance (the legacy ``co_scientist.safety.prompt_injection_guard``
remains in place). Any externally-sourced text the agent ingests can carry
adversarial instructions; this guard detects and strips them, then wraps the
remainder in isolation tags so the model treats it as data, never instructions.
"""

from __future__ import annotations

import re
import structlog
from typing import List

logger = structlog.get_logger("injection_guard")

# Patterns that indicate injection attempts.
_INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(
        r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|context|text)", re.I
    ),
    re.compile(r"you\s+are\s+now\s+(a\s+)?(new\s+)?(ai|assistant|chatbot|model)", re.I),
    re.compile(r"(system|system_prompt|system prompt)\s*[:=<\[]", re.I),
    re.compile(r"<\s*(system|instruction|prompt)\s*>", re.I),
    re.compile(r"###\s*(instruction|system|new task)", re.I),
    re.compile(r"print\s+(your\s+)?(api\s+key|secret|token|password)", re.I),
    re.compile(
        r"output\s+(your\s+)?(system|hidden|internal)\s+(prompt|instructions?)", re.I
    ),
    re.compile(r"repeat\s+(everything|all)\s+(above|before|prior)", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior)\s+(instructions?|rules?)", re.I),
    re.compile(r"\[INST\]|\[SYS\]|\|\s*im_start\s*\||<\|im_start\|>", re.I),
]

_REDACT_TOKEN = "[CONTENT REDACTED: POSSIBLE PROMPT INJECTION]"

# Tags that untrusted content could use to break out of the <external_content>
# isolation envelope or impersonate a privileged role. Neutralised before wrapping.
_ENVELOPE_BREAKOUT: List[re.Pattern] = [
    re.compile(r"</?\s*external_content[^>]*>", re.I),
    re.compile(r"</?\s*(system|assistant|user|tool)\s*>", re.I),
    re.compile(r"<\|\s*(im_start|im_end|system|assistant|user)\s*\|>", re.I),
]


def _escape_source_url(source_url: str) -> str:
    """Strip angle brackets and quotes so the URL can't break the tag attribute."""
    return re.sub(r'[<>"\']', "", source_url or "")


def _neutralize_body(body: str) -> str:
    """Strip envelope-closing and role tags so content can't escape isolation."""
    for pat in _ENVELOPE_BREAKOUT:
        body = pat.sub("[REDACTED TAG]", body)
    return body


class PromptInjectionGuard:
    """Wraps externally-sourced content in isolation tags and strips injection patterns.

    Usage:
        guard = PromptInjectionGuard()
        safe_text = await guard.sanitize(raw_web_content)
    """

    def __init__(self, strict: bool = True):
        """strict=True redacts matching lines; strict=False only logs (for debugging)."""
        self.strict = strict
        self._detected_count = 0

    async def sanitize(self, content: str, source_url: str = "") -> str:
        """Sanitize web content before LLM ingestion; returns isolated, stripped text."""
        if not content:
            return content

        cleaned_lines: List[str] = []
        injection_found = False
        for line in content.split("\n"):
            if self._is_injection(line):
                injection_found = True
                self._detected_count += 1
                logger.warning(
                    "injection_detected",
                    source=source_url or "unknown",
                    line=line[:100],
                )
                if self.strict:
                    cleaned_lines.append(_REDACT_TOKEN)
                    continue
            cleaned_lines.append(line)

        cleaned = _neutralize_body("\n".join(cleaned_lines))
        safe_source = _escape_source_url(source_url)
        isolated = (
            f'<external_content source="{safe_source}">\n{cleaned}\n</external_content>'
        )
        if injection_found:
            isolated = f"[WARNING: Injection attempts detected and redacted from this source]\n{isolated}"
        return isolated

    def _is_injection(self, text: str) -> bool:
        return any(p.search(text) for p in _INJECTION_PATTERNS)

    @property
    def detection_count(self) -> int:
        return self._detected_count
