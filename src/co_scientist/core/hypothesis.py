"""
Hypothesis data models and management for the Co-Scientist framework.

This module defines the core Hypothesis entity that flows through the entire
multi-agent pipeline. It is generic over the candidate domain model.
"""

from __future__ import annotations

import uuid
import time
import json
from enum import Enum
from typing import Any, Dict, List, Optional, TypeVar, Generic, Type
from datetime import datetime, timezone

def _now():
    return datetime.now(timezone.utc)
from pydantic import BaseModel, Field

try:
    from co_scientist.core.database import HypothesisModel
    HAS_DB = True
except ImportError:
    HAS_DB = False

import numpy as np

from co_scientist.ranking.trueskill_adapter import TrueSkillRating

CandidateT = TypeVar("CandidateT", bound=BaseModel)

class HypothesisStatus(str, Enum):
    """Lifecycle status of a hypothesis through the agent pipeline."""
    GENERATED = "generated"
    UNDER_REVIEW = "under_review"
    REFLECTED = "reflected"
    CLUSTERED = "clustered"
    EVOLVED = "evolved"
    RANKED = "ranked"
    VALIDATED = "validated"
    CONSENSUS_PASSED = "consensus_passed"
    REJECTED = "rejected"
    SELECTED = "selected"
    EXPERIMENT_DESIGNED = "experiment_designed"
    TESTED = "tested"

class Evidence(BaseModel):
    """Evidence supporting or refuting a hypothesis."""
    source: str
    evidence_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    supports: bool = True
    details: str = ""
    relevance_score: float = 0.5
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Criticism(BaseModel):
    """Critical evaluation from the Reflection agent."""
    criticism_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    critic_agent: str = ""
    dimension: str
    severity: str
    content: str
    evidence: Optional[str] = None
    suggested_fix: Optional[str] = None
    verified: bool = False

class ProximityVector(BaseModel):
    """Vector representation for proximity analysis."""
    embedding: List[float] = Field(default_factory=list)
    embedding_model: str = "default"
    dimension: int = 0
    cluster_id: int = -1
    similarity_to_centroid: float = 0.0

    def cosine_similarity(self, other: ProximityVector) -> float:
        if not self.embedding or not other.embedding:
            return 0.0
        a = np.array(self.embedding)
        b = np.array(other.embedding)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

class Hypothesis(BaseModel, Generic[CandidateT]):
    """Central hypothesis entity in the Co-Scientist framework."""
    # Identity
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    version: int = 1
    parent_ids: List[str] = Field(default_factory=list)
    source_agent: str = ""
    cycle_number: int = 0
    generation: int = 0

    # Content
    title: str = ""
    summary: str = ""
    rationale: str = ""
    research_goal: str = ""
    testable_predictions: List[str] = Field(default_factory=list)
    
    # Generic Domain Models
    candidates: List[CandidateT] = Field(default_factory=list)

    # Status tracking
    status: HypothesisStatus = HypothesisStatus.GENERATED
    pipeline_stage: str = "generation"

    # Quality metrics
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # Agent outputs
    criticisms: List[Criticism] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    proximity: ProximityVector = Field(default_factory=ProximityVector)

    # Ranking
    trueskill_rating: TrueSkillRating = Field(default_factory=TrueSkillRating)
    debate_wins: int = 0
    debate_losses: int = 0

    # Metadata
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def add_criticism(self, criticism: Criticism) -> None:
        self.criticisms.append(criticism)
        self.updated_at = _now()

    def add_evidence(self, ev: Evidence) -> None:
        self.evidence.append(ev)
        self.updated_at = _now()
        
    def update_rating(self, mu: float, sigma: float, won: bool = False, draw: bool = False) -> None:
        self.trueskill_rating = TrueSkillRating(mu=mu, sigma=sigma)
        if won:
            self.debate_wins += 1
        elif not draw:
            self.debate_losses += 1
        self.updated_at = _now()

    async def save(self, db_manager: Any = None) -> None:
        # Saving handled by Orchestrator or DB manager now
        pass

class HypothesisPool(Generic[CandidateT]):
    """Manages a collection of generic hypotheses."""

    def __init__(self, candidate_model: str):
        self.hypotheses: Dict[str, Hypothesis[CandidateT]] = {}
        self.candidate_model_name = candidate_model

    @classmethod
    def from_json(cls, json_str: str, candidate_model: Type[CandidateT]) -> HypothesisPool[CandidateT]:
        data = json.loads(json_str)
        pool = cls(candidate_model=candidate_model.__name__)
        for item in data.get("hypotheses", []):
            h = Hypothesis[candidate_model](**item)
            pool.hypotheses[h.id] = h
        return pool

    def add(self, hypothesis: Hypothesis[CandidateT]) -> None:
        self.hypotheses[hypothesis.id] = hypothesis

    def get(self, hypothesis_id: str) -> Optional[Hypothesis[CandidateT]]:
        return self.hypotheses.get(hypothesis_id)

    def get_by_status(self, status: HypothesisStatus) -> List[Hypothesis[CandidateT]]:
        return [h for h in self.hypotheses.values() if h.status == status]

    def get_leaderboard(self, sort_by: str = "conservative") -> List[Dict[str, Any]]:
        # Sort by epistemic optionally
        def sort_key(h):
            if sort_by == "epistemic":
                # Assuming candidate has epistemic score
                uncertainties = [getattr(c, "epistemic_uncertainty_score", 0.0) for c in h.candidates]
                max_u = max(uncertainties) if uncertainties else 0.0
                return h.trueskill_rating.conservative_rating - (max_u * 10)
            return h.trueskill_rating.conservative_rating

        ranked = sorted(self.hypotheses.values(), key=sort_key, reverse=True)
        return [
            {
                "id": h.id,
                "title": h.title[:80],
                "mu": round(h.trueskill_rating.mu, 2),
                "sigma": round(h.trueskill_rating.sigma, 2),
                "wins": h.debate_wins,
                "losses": h.debate_losses,
                "status": h.status.value,
            }
            for h in ranked
        ]

    def __len__(self) -> int:
        return len(self.hypotheses)

    def __iter__(self):
        return iter(self.hypotheses.values())
