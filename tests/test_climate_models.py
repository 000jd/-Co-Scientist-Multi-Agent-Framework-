"""
Tests for Climate Models and Generic Hypothesis.
"""

import pytest
from co_scientist.core.hypothesis import Hypothesis, HypothesisPool, TrueSkillRating
from co_scientist.core.climate_models import ClimateIntervention, InterventionCategory, RiskMatrix, RiskDimension, Reversibility
from co_scientist.ranking.trueskill_adapter import CombinedLeaderboard
import json

def get_dummy_intervention() -> ClimateIntervention:
    risk_matrix = RiskMatrix(
        ecological=RiskDimension(score=0.2, description="Low ecological risk"),
        economic=RiskDimension(score=0.4, description="Moderate economic risk"),
        social_equity=RiskDimension(score=0.1, description="Low social equity risk"),
        geopolitical=RiskDimension(score=0.3, description="Low geopolitical risk"),
    )
    return ClimateIntervention(
        name="Ocean Alkalinity Enhancement",
        description="Adding crushed olivine to oceans.",
        category=InterventionCategory.GEOENGINEERING_CDR,
        technology_readiness_level=4,
        risk_assessment=risk_matrix,
        reversibility=Reversibility.LOW,
        epistemic_uncertainty_score=0.5
    )

def test_climate_intervention_uncertainty():
    candidate = get_dummy_intervention()
    assert candidate.epistemic_uncertainty_score == 0.5
    candidate.increase_uncertainty("test_reason", 0.3)
    assert candidate.epistemic_uncertainty_score == 0.8
    candidate.decrease_uncertainty("convergent_evidence", 0.1)
    assert round(candidate.epistemic_uncertainty_score, 1) == 0.7

def test_generic_hypothesis_serialization():
    candidate = get_dummy_intervention()
    hypothesis = Hypothesis[ClimateIntervention](
        title="Test Hypothesis",
        statement="Test Statement",
        candidates=[candidate]
    )
    
    assert hypothesis.candidates[0].name == "Ocean Alkalinity Enhancement"
    
    pool = HypothesisPool[ClimateIntervention](candidate_model=ClimateIntervention.__name__)
    pool.add(hypothesis)
    
    # Mocking serialization/deserialization payload
    pool_json = json.dumps({"hypotheses": [json.loads(h.model_dump_json()) for h in pool.hypotheses.values()]})
    
    restored_pool = HypothesisPool.from_json(pool_json, ClimateIntervention)
    assert len(restored_pool) == 1
    restored_hyp = restored_pool.get(hypothesis.id)
    assert restored_hyp is not None
    assert isinstance(restored_hyp.candidates[0], ClimateIntervention)
    assert restored_hyp.candidates[0].category == InterventionCategory.GEOENGINEERING_CDR

def test_combined_leaderboard():
    leaderboard = CombinedLeaderboard()
    leaderboard.add_or_update(
        hypothesis_id="h1",
        title="High Uncertainty",
        trueskill_mu=25.0,
        trueskill_sigma=2.0,
        debate_wins=5,
        debate_losses=1,
        epistemic_uncertainty_score=0.8
    )
    
    leaderboard.add_or_update(
        hypothesis_id="h2",
        title="Low Uncertainty",
        trueskill_mu=25.0,
        trueskill_sigma=2.0,
        debate_wins=5,
        debate_losses=1,
        epistemic_uncertainty_score=0.2
    )
    
    display = leaderboard.to_display(sort_by="adjusted")
    assert len(display) == 2
    assert display[0]["id"] == "h2"  # h2 should rank higher due to lower uncertainty penalty
    assert display[1]["id"] == "h1"
