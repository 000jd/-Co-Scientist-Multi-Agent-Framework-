"""SimulationValidator + ConsensusValidator — real closed-loop checks (CLAUDE.md §5 Bug 5).

Both were stubs that unconditionally returned ``{"status":"validated","evidence":[]}``.
SimulationValidator now assesses computational/mechanistic plausibility; ConsensusValidator
aggregates the evidence accumulated on the hypothesis into a real consensus verdict that
can fail.
"""

from __future__ import annotations

from typing import Any, Dict

import structlog
from pydantic import BaseModel

from .base import BaseValidator
from .literature import (
    _avg_uncertainty,
    _describe,
    _heuristic,
    _verdict_to_result,
    _LitVerdict,
)
from co_scientist.core.hypothesis import Evidence, Hypothesis, HypothesisStatus

log = structlog.get_logger("validation.sandbox")


class SimulationValidator(BaseValidator):
    async def validate(self, hypothesis: Hypothesis) -> Dict[str, Any]:
        if self.llm_router is not None:
            try:
                verdict = await self.llm_router.complete(
                    [
                        {
                            "role": "system",
                            "content": "You are a simulation/mechanism reviewer. Judge "
                            "whether the hypothesis's predicted mechanism is internally consistent and "
                            'plausibly reproducible in silico. Reply JSON {"supported": bool, '
                            '"confidence": 0..1, "rationale": str, "key_references": [str]}.',
                        },
                        {"role": "user", "content": _describe(hypothesis)},
                    ],
                    response_format=_LitVerdict,
                )
                if isinstance(verdict, _LitVerdict):
                    return _verdict_to_result(verdict, source="simulation_llm")
            except Exception as e:  # noqa: BLE001
                log.warning("simulation.llm_failed", error=str(e))
        return _heuristic(hypothesis, channel="simulation")


class ConsensusValidator(BaseValidator):
    """Aggregates accumulated Evidence into a consensus verdict (can fail)."""

    async def validate(self, hypothesis: Hypothesis) -> Dict[str, Any]:
        evidence = list(hypothesis.evidence)
        if not evidence:
            # nothing to reach consensus over yet
            unc = _avg_uncertainty(hypothesis)
            status = (
                HypothesisStatus.UNDER_REVIEW
                if unc <= 0.7
                else HypothesisStatus.REJECTED
            )
            return {
                "status": status.value,
                "evidence": [],
                "confidence": 1.0 - unc,
                "n_evidence": 0,
            }

        supporting = [e for e in evidence if e.supports]
        refuting = [e for e in evidence if not e.supports]
        support_conf = sum(e.confidence for e in supporting)
        refute_conf = sum(e.confidence for e in refuting)
        total = support_conf + refute_conf or 1.0
        agreement = support_conf / total

        if agreement >= 0.66:
            status = HypothesisStatus.CONSENSUS_PASSED
        elif agreement <= 0.34:
            status = HypothesisStatus.REJECTED
        else:
            status = HypothesisStatus.UNDER_REVIEW

        summary = Evidence(
            source="consensus",
            evidence_type="consensus",
            confidence=round(max(agreement, 1 - agreement), 3),
            supports=agreement >= 0.5,
            details=f"{len(supporting)} supporting / {len(refuting)} refuting; agreement={agreement:.2f}",
        )
        return {
            "status": status.value,
            "evidence": [summary.model_dump()],
            "confidence": round(agreement, 3),
            "n_evidence": len(evidence),
        }
