from abc import ABC, abstractmethod
from typing import Dict, Any
from co_scientist.core.hypothesis import Hypothesis

class BaseValidator(ABC):
    def __init__(self, config, event_bus, llm_router):
        self.config = config
        self.event_bus = event_bus
        self.llm_router = llm_router
        self.memory = None

    @abstractmethod
    async def validate(self, hypothesis: Hypothesis) -> Dict[str, Any]:
        pass
