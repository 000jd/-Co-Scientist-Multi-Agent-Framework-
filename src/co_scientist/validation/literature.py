"""LiteratureValidator — real closed-loop literature check (CLAUDE.md §5 Bug 5).

Was a stub that unconditionally returned ``{"status":"validated","evidence":[]}``.
Now it actually inspects the hypothesis and can return REJECTED / UNDER_REVIEW —
the precondition for a meaningful Verify stage.
"""

from __future__ import annotations

from typing import Any, Dict, List

import structlog
from pydantic import BaseModel

from .base import BaseValidator
from co_scientist.core.hypothesis import Evidence, Hypothesis, HypothesisStatus

log = structlog.get_logger("validation.literature")


class _LitVerdict(BaseModel):
    supported: bool
    confidence: float
    rationale: str
    key_references: List[str] = []


def _avg_uncertainty(hypothesis: Hypothesis) -> float:
    scores = [
        getattr(c, "epistemic_uncertainty_score", 0.5) for c in hypothesis.candidates
    ]
    return sum(scores) / len(scores) if scores else 0.5


class LiteratureValidator(BaseValidator):
    async def validate(self, hypothesis: Hypothesis) -> Dict[str, Any]:
        if self.llm_router is not None:
            try:
                verdict = await self.llm_router.complete(
                    [
                        {
                            "role": "system",
                            "content": "You are a scientific literature reviewer. Judge "
                            "whether the hypothesis is consistent with and supported by established "
                            'literature. Reply JSON {"supported": bool, "confidence": 0..1, '
                            '"rationale": str, "key_references": [str]}. Be willing to say unsupported.',
                        },
                        {"role": "user", "content": _describe(hypothesis)},
                    ],
                    response_format=_LitVerdict,
                )
                if isinstance(verdict, _LitVerdict):
                    return _verdict_to_result(verdict, source="literature_llm")
            except Exception as e:  # noqa: BLE001
                log.warning("literature.llm_failed", error=str(e))

        return _heuristic(hypothesis, channel="literature")


def _describe(h: Hypothesis) -> str:
    preds = "; ".join(h.testable_predictions) or "(none)"
    return f"TITLE: {h.title}\nSUMMARY: {h.summary}\nRATIONALE: {h.rationale}\nPREDICTIONS: {preds}"


def _verdict_to_result(v: _LitVerdict, source: str) -> Dict[str, Any]:
    if v.supported and v.confidence >= 0.6:
        status = HypothesisStatus.VALIDATED
    elif not v.supported and v.confidence >= 0.6:
        status = HypothesisStatus.REJECTED
    else:
        status = HypothesisStatus.UNDER_REVIEW
    evidence = Evidence(
        source=source,
        evidence_type="literature",
        confidence=v.confidence,
        supports=v.supported,
        details=v.rationale,
        metadata={"references": v.key_references},
    )
    return {
        "status": status.value,
        "evidence": [evidence.model_dump()],
        "confidence": v.confidence,
    }


def _heuristic(hypothesis: Hypothesis, channel: str) -> Dict[str, Any]:
    unc = _avg_uncertainty(hypothesis)
    has_pred = len(hypothesis.testable_predictions) > 0
    if unc < 0.4 and has_pred:
        status, conf, supports = HypothesisStatus.VALIDATED, 0.7, True
    elif unc > 0.7:
        status, conf, supports = HypothesisStatus.REJECTED, 0.6, False
    else:
        status, conf, supports = HypothesisStatus.UNDER_REVIEW, 0.5, True
    evidence = Evidence(
        source=f"{channel}_heuristic",
        evidence_type=channel,
        confidence=conf,
        supports=supports,
        details=f"avg_uncertainty={unc:.2f}, testable_predictions={has_pred}",
    )
    return {
        "status": status.value,
        "evidence": [evidence.model_dump()],
        "confidence": conf,
    }
