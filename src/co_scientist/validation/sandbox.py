from typing import Dict, Any
from .base import BaseValidator
from co_scientist.core.hypothesis import Hypothesis

class SimulationValidator(BaseValidator):
    async def validate(self, hypothesis: Hypothesis) -> Dict[str, Any]:
        return {"status": "validated", "evidence": []}

class ConsensusValidator(BaseValidator):
    async def validate(self, hypothesis: Hypothesis) -> Dict[str, Any]:
        return {"status": "validated", "evidence": []}
