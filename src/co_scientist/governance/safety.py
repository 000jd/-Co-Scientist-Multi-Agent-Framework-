"""Safety verdicts and checker ABC (adapted from co_scientist.safety.base).

Kept independent of the legacy module so governance is self-contained. The legacy
``co_scientist.safety.base`` remains in place for existing imports.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel

from co_scientist.core.domain_models import Candidate


class SafetyVerdict(str, Enum):
    APPROVED = "approved"
    CAUTION = "caution"
    FLAGGED = "flagged"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"


class SafetyFlag(BaseModel):
    severity: str
    description: str
    evidence_basis: str
    recommended_action: str


class SafetyReport(BaseModel):
    verdict: SafetyVerdict
    flags: List[SafetyFlag] = []


class SafetyChecker(ABC):
    @abstractmethod
    async def check(
        self, candidate: Candidate, context: Optional[Dict] = None
    ) -> SafetyReport: ...


class DefaultSafetyChecker(SafetyChecker):
    async def check(
        self, candidate: Candidate, context: Optional[Dict] = None
    ) -> SafetyReport:
        return SafetyReport(verdict=SafetyVerdict.APPROVED)
