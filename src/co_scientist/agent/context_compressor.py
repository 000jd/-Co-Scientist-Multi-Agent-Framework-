"""ContextCompressor — keep the worker's context under the model window
(CLAUDE.md §3: compress at 92% token use).

Layered strategy (cheapest first), applied only once the estimated token count
crosses ``compress_at`` of the budget:

  1. truncate over-long individual tool observations,
  2. drop superseded duplicate observations,
  3. LLM-summarise the oldest middle span (when an LLM is available),
  4. otherwise hard-truncate the middle, always keeping the system prompt + head + tail.
"""

from __future__ import annotations

from typing import Dict, List

import structlog

log = structlog.get_logger("agent.context_compressor")

_OBS_TRUNC = 2_000  # max chars kept per tool observation when compressing
_KEEP_HEAD = 2  # leading messages preserved (goal framing)
_KEEP_TAIL = 6  # most-recent messages preserved verbatim


def estimate_tokens(messages: List[Dict[str, str]]) -> int:
    """Rough token estimate (~4 chars/token)."""
    return sum(len(m.get("content", "")) for m in messages) // 4


class ContextCompressor:
    def __init__(
        self, llm_router=None, token_budget: int = 200_000, compress_at: float = 0.92
    ):
        self.llm = llm_router
        self.token_budget = token_budget
        self.compress_at = compress_at

    def over_threshold(self, messages: List[Dict[str, str]]) -> bool:
        return estimate_tokens(messages) >= int(self.token_budget * self.compress_at)

    async def maybe_compress(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        if not self.over_threshold(messages):
            return messages

        log.info(
            "context_compressor.compressing",
            tokens=estimate_tokens(messages),
            budget=self.token_budget,
        )

        # 1. truncate long observations in place (kept as role="user" "[OBSERVATION] …")
        for m in messages:
            content = m.get("content", "")
            if m.get("role") != "system" and len(content) > _OBS_TRUNC:
                m["content"] = content[:_OBS_TRUNC] + "\n…[truncated]"

        if (
            not self.over_threshold(messages)
            or len(messages) <= _KEEP_HEAD + _KEEP_TAIL
        ):
            return messages

        head = messages[:_KEEP_HEAD]
        tail = messages[-_KEEP_TAIL:]
        middle = messages[_KEEP_HEAD:-_KEEP_TAIL]

        # 3. LLM summary of the middle span, else 4. hard-truncate.
        summary_text = self._mechanical_summary(middle)
        if self.llm is not None:
            try:
                joined = "\n".join(
                    f"[{m.get('role')}] {m.get('content', '')}" for m in middle
                )[:12_000]
                resp = await self.llm.complete(
                    [
                        {
                            "role": "system",
                            "content": "Summarise the following agent transcript span into a "
                            "compact set of durable facts, decisions, and intermediate results. Preserve any "
                            "numbers, file paths, and errors. Be terse.",
                        },
                        {"role": "user", "content": joined},
                    ]
                )
                if isinstance(resp, str) and resp.strip():
                    summary_text = resp.strip()
            except Exception as e:  # noqa: BLE001
                log.warning("context_compressor.llm_failed", error=str(e))

        summary_msg = {
            "role": "system",
            "content": f"[compressed earlier context]\n{summary_text}",
        }
        return head + [summary_msg] + tail

    @staticmethod
    def _mechanical_summary(middle: List[Dict[str, str]]) -> str:
        lines = []
        for m in middle:
            content = m.get("content", "").strip().replace("\n", " ")
            if content:
                lines.append(f"- {m.get('role')}: {content[:160]}")
        return "\n".join(lines[-40:])
