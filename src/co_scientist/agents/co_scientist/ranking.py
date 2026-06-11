"""
Ranking Agent for Domain-Agnostic Hypotheses.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from co_scientist.agents.base import BaseAgent
from co_scientist.core.hypothesis import Hypothesis, HypothesisStatus
from co_scientist.ranking.trueskill_adapter import update_rating, TrueSkillRating, CombinedLeaderboard
from co_scientist.core.events import EventType

class DebateWinner(BaseModel):
    winner_id: Optional[str]
    rationale: str

class RankingAgent(BaseAgent):
    """Ranks hypotheses through tournament debates using TrueSkill."""
    agent_name = "ranking"
    
    def __init__(self, config, event_bus, llm_router, leaderboard: CombinedLeaderboard):
        super().__init__(config, event_bus, llm_router)
        self.leaderboard = leaderboard
        
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        hypotheses: List[Hypothesis] = context["hypotheses"]
        
        # 1. Generate debate pairs
        pairs = self._generate_pairs(hypotheses)
        
        # 2. Run debates
        import asyncio
        debate_coros = [self._run_debate(h1, h2) for h1, h2 in pairs]
        winners = await asyncio.gather(*debate_coros, return_exceptions=True)
        
        for (h1, h2), winner_id in zip(pairs, winners):
            if isinstance(winner_id, Exception):
                winner_id = None
                
            ts1 = TrueSkillRating(mu=h1.trueskill_rating.mu, sigma=h1.trueskill_rating.sigma)
            ts2 = TrueSkillRating(mu=h2.trueskill_rating.mu, sigma=h2.trueskill_rating.sigma)
            
            if winner_id == h1.id:
                w_new, l_new = update_rating(ts1, ts2)
                h1.update_rating(w_new.mu, w_new.sigma, won=True)
                h2.update_rating(l_new.mu, l_new.sigma, won=False)
            elif winner_id == h2.id:
                w_new, l_new = update_rating(ts2, ts1)
                h2.update_rating(w_new.mu, w_new.sigma, won=True)
                h1.update_rating(l_new.mu, l_new.sigma, won=False)
            else:
                r1_new, r2_new = update_rating(ts1, ts2, draw=True)
                h1.update_rating(r1_new.mu, r1_new.sigma, won=False, draw=True)
                h2.update_rating(r2_new.mu, r2_new.sigma, won=False, draw=True)
                
        # 3. Update CombinedLeaderboard
        for h in hypotheses:
            max_uncertainty = 0.5
            for c in h.candidates:
                if hasattr(c, "epistemic_uncertainty_score"):
                    max_uncertainty = max(max_uncertainty, c.epistemic_uncertainty_score)
                    
            self.leaderboard.add_or_update(
                hypothesis_id=h.id,
                title=h.title,
                trueskill_mu=h.trueskill_rating.mu,
                trueskill_sigma=h.trueskill_rating.sigma,
                debate_wins=h.debate_wins,
                debate_losses=h.debate_losses,
                epistemic_uncertainty_score=max_uncertainty
            )
            
        # 4. Flag high-ranking high-uncertainty
        flagged = self.leaderboard.get_flagged()
        if flagged:
            await self._publish_event(
                EventType.SAFETY_FLAG_RAISED,
                {"flagged_hypotheses": [f.hypothesis_id for f in flagged]}
            )
            
        for h in hypotheses:
            h.status = HypothesisStatus.RANKED
            
        return {
            "hypotheses": hypotheses,
            "leaderboard": self.leaderboard.to_display(),
            "flagged": flagged,
        }
        
    def _generate_pairs(self, hypotheses: List[Hypothesis], max_debates: int = 30) -> List[tuple]:
        import random
        pairs = []
        # Round-robin within skill-tiers (banded TrueSkill)
        sorted_h = sorted(hypotheses, key=lambda h: h.trueskill_rating.conservative_rating)
        
        # Band into groups of ~5 for local tournaments
        band_size = 5
        for i in range(0, len(sorted_h), band_size):
            band = sorted_h[i:i+band_size]
            for a in range(len(band)):
                for b in range(a+1, len(band)):
                    pairs.append((band[a], band[b]))
        
        # A few cross-band matches for global ranking
        if len(sorted_h) > band_size:
            for _ in range(min(10, max_debates - len(pairs))):
                pairs.append((random.choice(sorted_h), random.choice(sorted_h)))
        
        return pairs[:max_debates]
        
    async def _run_debate(self, h1: Hypothesis, h2: Hypothesis) -> Optional[str]:
        prompt = f"""You are a senior domain expert judging a debate between two proposed hypotheses in the domain: {self.config.domain.name}.
        
Hypothesis A (ID: {h1.id}):
Title: {h1.title}
Summary: {h1.summary}

Hypothesis B (ID: {h2.id}):
Title: {h2.title}
Summary: {h2.summary}

Evaluate both based on:
1. Impact
2. Feasibility
3. Novelty
4. Cost
5. Safety
6. Scalability

Select the winner. If it's a tie, return null for winner_id.
Provide a concise rationale for your decision."""

        try:
            response = await self._call_llm(
                system_prompt="You are an expert judge. Be objective and prioritize impactful, feasible, and safe hypotheses.",
                user_prompt=prompt,
                response_format=DebateWinner
            )
            return response.winner_id
        except Exception as e:
            # Fallback to draw if LLM fails
            return None
