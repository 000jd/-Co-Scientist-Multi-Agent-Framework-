"""
PromptInjectionGuard — sanitizes web/document content before LLM ingestion.

Kimi Work threat model: any page the agent browses can contain adversarial
instructions targeting the agent. This guard detects and strips them.

Pattern: wrap ALL externally-sourced text with isolation tags + detection pass.
"""

import re
import logging
from typing import List

logger = logging.getLogger("injection_guard")

# Patterns that indicate injection attempts
_INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|context|text)", re.I),
    re.compile(r"you\s+are\s+now\s+(a\s+)?(new\s+)?(ai|assistant|chatbot|model)", re.I),
    re.compile(r"(system|system_prompt|system prompt)\s*[:=<\[]", re.I),
    re.compile(r"<\s*(system|instruction|prompt)\s*>", re.I),
    re.compile(r"###\s*(instruction|system|new task)", re.I),
    re.compile(r"print\s+(your\s+)?(api\s+key|secret|token|password)", re.I),
    re.compile(r"output\s+(your\s+)?(system|hidden|internal)\s+(prompt|instructions?)", re.I),
    re.compile(r"repeat\s+(everything|all)\s+(above|before|prior)", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior)\s+(instructions?|rules?)", re.I),
    re.compile(r"\[INST\]|\[SYS\]|\|\s*im_start\s*\||<\|im_start\|>", re.I),
]

# Replacement for detected segments
_REDACT_TOKEN = "[CONTENT REDACTED: POSSIBLE PROMPT INJECTION]"


class PromptInjectionGuard:
    """
    Wraps externally-sourced content in isolation tags and strips injection patterns.
    
    Usage:
        guard = PromptInjectionGuard()
        safe_text = await guard.sanitize(raw_web_content)
        
    The safe_text is then safe to embed in LLM prompts.
    """
    
    def __init__(self, strict: bool = True):
        """
        strict=True: redact any line matching injection patterns.
        strict=False: log warning only, don't redact (for debugging).
        """
        self.strict = strict
        self._detected_count = 0
        
    async def sanitize(self, content: str, source_url: str = "") -> str:
        """
        Sanitize web content before LLM ingestion.
        Returns content wrapped in isolation markers with injections stripped.
        """
        if not content:
            return content
            
        lines = content.split("\n")
        cleaned_lines = []
        injection_found = False
        
        for line in lines:
            if self._is_injection(line):
                injection_found = True
                self._detected_count += 1
                logger.warning(f"Injection attempt detected from {source_url or 'unknown'}: {line[:100]}")
                if self.strict:
                    cleaned_lines.append(_REDACT_TOKEN)
                    continue
            cleaned_lines.append(line)
            
        cleaned = "\n".join(cleaned_lines)
        
        # Wrap in isolation tags — the outer LLM system prompt should instruct the model
        # to treat content inside <external_content> as data, never as instructions
        isolated = (
            f"<external_content source=\"{source_url}\">\n"
            f"{cleaned}\n"
            f"</external_content>"
        )
        
        if injection_found:
            isolated = f"[WARNING: Injection attempts detected and redacted from this source]\n{isolated}"
            
        return isolated
        
    def _is_injection(self, text: str) -> bool:
        return any(p.search(text) for p in _INJECTION_PATTERNS)
        
    @property
    def detection_count(self) -> int:
        return self._detected_count
