from abc import ABC, abstractmethod
from enum import Enum
from pydantic import BaseModel
from typing import List, Dict, Any
from co_scientist.core.domain_models import Candidate

class SafetyVerdict(Enum):
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
    async def check(self, candidate: Candidate, context: Dict = None) -> SafetyReport:
        pass

class DefaultSafetyChecker(SafetyChecker):
    async def check(self, candidate: Candidate, context: Dict = None) -> SafetyReport:
        return SafetyReport(verdict=SafetyVerdict.APPROVED)
