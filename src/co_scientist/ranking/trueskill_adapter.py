"""
TrueSkill ranking adapter with CombinedLeaderboard incorporating epistemic uncertainty.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import time
from trueskill import Rating, rate_1vs1, setup

# Setup Trueskill environment
setup(mu=25.0, sigma=8.333, beta=4.167, tau=0.0833, draw_probability=0.1)

class TrueSkillRating(BaseModel):
    """Pure TrueSkill tracking for debate win/loss."""
    mu: float = 25.0
    sigma: float = 8.333
    
    @property
    def conservative_rating(self) -> float:
        return self.mu - 3 * self.sigma
        
    @property
    def skill_estimate(self) -> float:
        return self.mu

def update_rating(
    winner: TrueSkillRating,
    loser: TrueSkillRating,
    draw: bool = False,
) -> tuple[TrueSkillRating, TrueSkillRating]:
    """Update ratings after a debate.
    
    Args:
        winner: Rating of the debate winner
        loser: Rating of the debate loser
        draw: If True, treat as a draw (winner/loser labels don't matter)
    
    Returns:
        (updated_winner_rating, updated_loser_rating)
    """
    t1 = Rating(mu=winner.mu, sigma=winner.sigma)
    t2 = Rating(mu=loser.mu, sigma=loser.sigma)
    
    if draw:
        t1_new, t2_new = rate_1vs1(t1, t2, drawn=True)
    else:
        t1_new, t2_new = rate_1vs1(t1, t2)  # t1 wins
        
    return (
        TrueSkillRating(mu=t1_new.mu, sigma=t1_new.sigma),
        TrueSkillRating(mu=t2_new.mu, sigma=t2_new.sigma),
    )

class CombinedLeaderboardEntry(BaseModel):
    hypothesis_id: str
    title: str
    trueskill_mu: float
    trueskill_sigma: float
    debate_wins: int
    debate_losses: int
    epistemic_uncertainty_score: float
    
    @property
    def trueskill_conservative(self) -> float:
        return self.trueskill_mu - 3 * self.trueskill_sigma
        
    @property
    def adjusted_ranking(self) -> float:
        # Penalizes high uncertainty
        return self.trueskill_conservative - (self.epistemic_uncertainty_score * 10.0)
        
    @property
    def confidence_flag(self) -> str:
        if self.epistemic_uncertainty_score > 0.7 and self.trueskill_conservative > 15.0:
            return "HIGH_RANKING_HIGH_UNCERTAINTY"
        elif self.epistemic_uncertainty_score > 0.5:
            return "MODERATE_UNCERTAINTY"
        elif self.epistemic_uncertainty_score < 0.3 and self.trueskill_conservative > 15.0:
            return "WELL_ESTABLISHED"
        return "NOMINAL"

class CombinedLeaderboard:
    def __init__(self):
        self.entries: Dict[str, CombinedLeaderboardEntry] = {}
        
    def add_or_update(
        self,
        hypothesis_id: str,
        title: str,
        trueskill_mu: float,
        trueskill_sigma: float,
        debate_wins: int,
        debate_losses: int,
        epistemic_uncertainty_score: float
    ):
        self.entries[hypothesis_id] = CombinedLeaderboardEntry(
            hypothesis_id=hypothesis_id,
            title=title,
            trueskill_mu=trueskill_mu,
            trueskill_sigma=trueskill_sigma,
            debate_wins=debate_wins,
            debate_losses=debate_losses,
            epistemic_uncertainty_score=epistemic_uncertainty_score
        )
        
    _FLAGGED_STATUSES = {"HIGH_RANKING_HIGH_UNCERTAINTY", "MODERATE_UNCERTAINTY"}

    def get_flagged(self) -> List[CombinedLeaderboardEntry]:
        return [e for e in self.entries.values() if e.confidence_flag in self._FLAGGED_STATUSES]
        
    def to_display(self, sort_by: str = "adjusted") -> List[Dict[str, Any]]:
        if sort_by == "adjusted":
            sorted_entries = sorted(self.entries.values(), key=lambda e: e.adjusted_ranking, reverse=True)
        elif sort_by == "trueskill":
            sorted_entries = sorted(self.entries.values(), key=lambda e: e.trueskill_conservative, reverse=True)
        else:
            sorted_entries = list(self.entries.values())
            
        return [
            {
                "rank": i + 1,
                "id": e.hypothesis_id,
                "title": e.title,
                "elo": round(e.trueskill_conservative, 1),
                "adjusted_score": round(e.adjusted_ranking, 2),
                "mu": round(e.trueskill_mu, 2),
                "sigma": round(e.trueskill_sigma, 2),
                "uncertainty": round(e.epistemic_uncertainty_score, 3),
                "wins": e.debate_wins,
                "losses": e.debate_losses,
                "draws": 0,
                "tier": e.confidence_flag,
                "flag": e.confidence_flag,
            }
            for i, e in enumerate(sorted_entries)
        ]
