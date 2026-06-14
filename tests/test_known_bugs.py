"""Regression tests for CLAUDE.md §5 known bugs.

Bugs 1-3 were already fixed in the tree before this migration; these tests lock the
fixes (each would fail against the spec's "WRONG" code). Bug 5 (validation stubs) was
genuinely open and is fixed here. Bug 4 targets cli/tui.py, which does not exist.
"""

import pytest

from co_scientist.core.task_manager import ConvergenceTracker
from co_scientist.ranking.trueskill_adapter import CombinedLeaderboard
from co_scientist.core.hypothesis import Hypothesis, HypothesisPool, HypothesisStatus
from co_scientist.core.domain_models import Candidate


# ── Bug 1: ConvergenceTracker.is_converged must not crash on a short history ──
def test_bug1_convergence_no_index_error_at_window_boundary():
    t = ConvergenceTracker(window_size=3)
    for s in [
        1.0,
        2.0,
        3.0,
    ]:  # exactly window_size: the WRONG slice max([]) would raise
        t.record([{"id": "x", "adjusted_score": s}])
    assert t.is_converged is False  # improving + short history


def test_bug1_convergence_detects_flat():
    t = ConvergenceTracker(window_size=3)
    for s in [5.0, 5.0, 5.0, 5.0]:
        t.record([{"id": "x", "adjusted_score": s}])
    assert t.is_converged is True


# ── Bug 2: get_flagged returns only flagged entries, not everything ───────────
def test_bug2_get_flagged_filters():
    lb = CombinedLeaderboard()
    lb.add_or_update(
        "flagged",
        "moderate uncertainty",
        25.0,
        2.0,
        5,
        1,
        epistemic_uncertainty_score=0.6,
    )
    lb.add_or_update(
        "clean", "well established", 25.0, 2.0, 5, 1, epistemic_uncertainty_score=0.2
    )
    flagged_ids = {e.hypothesis_id for e in lb.get_flagged()}
    assert flagged_ids == {"flagged"}  # NOT both


# ── Bug 3: leaderboards expose adjusted_score (else convergence compares vs 0) ─
def test_bug3_combined_leaderboard_has_adjusted_score():
    lb = CombinedLeaderboard()
    lb.add_or_update("h", "t", 25.0, 2.0, 1, 0, epistemic_uncertainty_score=0.3)
    assert "adjusted_score" in lb.to_display()[0]


def test_bug3_hypothesis_pool_leaderboard_has_adjusted_score():
    pool = HypothesisPool[Candidate](candidate_model="Candidate")
    pool.add(
        Hypothesis[Candidate](
            title="h", candidates=[Candidate(name="c", description="d")]
        )
    )
    assert "adjusted_score" in pool.get_leaderboard()[0]


# ── Bug 5: validators do real work and can return non-"validated" ────────────
def _hyp(uncertainty: float, predictions=None):
    return Hypothesis[Candidate](
        title="t",
        summary="s",
        rationale="r",
        testable_predictions=predictions or [],
        candidates=[
            Candidate(
                name="c", description="d", epistemic_uncertainty_score=uncertainty
            )
        ],
    )


@pytest.mark.asyncio
async def test_bug5_literature_validator_can_reject():
    from co_scientist.validation.literature import LiteratureValidator

    v = LiteratureValidator(config=None, event_bus=None, llm_router=None)
    result = await v.validate(_hyp(uncertainty=0.9))
    assert result["status"] == HypothesisStatus.REJECTED.value
    assert result != {"status": "validated", "evidence": []}  # the old stub
    assert result["evidence"]  # real evidence attached


@pytest.mark.asyncio
async def test_bug5_literature_validator_can_validate():
    from co_scientist.validation.literature import LiteratureValidator

    v = LiteratureValidator(config=None, event_bus=None, llm_router=None)
    result = await v.validate(_hyp(uncertainty=0.2, predictions=["p1"]))
    assert result["status"] == HypothesisStatus.VALIDATED.value


@pytest.mark.asyncio
async def test_bug5_consensus_validator_aggregates_evidence():
    from co_scientist.validation.sandbox import ConsensusValidator
    from co_scientist.core.hypothesis import Evidence

    v = ConsensusValidator(config=None, event_bus=None, llm_router=None)

    h = _hyp(uncertainty=0.5)
    h.evidence = [
        Evidence(source="a", evidence_type="lit", confidence=0.8, supports=True),
        Evidence(source="b", evidence_type="sim", confidence=0.7, supports=True),
    ]
    result = await v.validate(h)
    assert result["status"] == HypothesisStatus.CONSENSUS_PASSED.value

    h.evidence = [
        Evidence(source="c", evidence_type="lit", confidence=0.9, supports=False)
    ]
    rejected = await v.validate(h)
    assert rejected["status"] == HypothesisStatus.REJECTED.value
