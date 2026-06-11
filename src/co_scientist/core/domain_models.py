"""
Domain models for generic candidate hypotheses.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from enum import Enum

class Candidate(BaseModel):
    """Generic candidate hypothesis."""
    name: str
    description: str
    domain_type: str = "generic"
    
    # Domain-agnostic fields (keep from original)
    score: float = 0.0
    epistemic_uncertainty_score: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Flexible metadata for any domain
    metrics: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)
    
    def increase_uncertainty(self, reason: str, amount: float) -> None:
        self.epistemic_uncertainty_score = min(1.0, self.epistemic_uncertainty_score + amount)
    
    def decrease_uncertainty(self, reason: str, amount: float) -> None:
        self.epistemic_uncertainty_score = max(0.0, self.epistemic_uncertainty_score - amount)
