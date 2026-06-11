"""
Base Agent class for Co-Scientist framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type
import logging
from pydantic import BaseModel

from co_scientist.core.config import Config
from co_scientist.core.events import EventBus, EventType, Event
from co_scientist.infrastructure.llm import LLMRouter
from co_scientist.memory.context_memory import ContextMemory

class BaseAgent(ABC):
    """Abstract base class for all Co-Scientist agents."""
    
    agent_name: str
    
    def __init__(
        self,
        config: Config,
        event_bus: EventBus,
        llm_router: LLMRouter,
    ):
        self.config = config
        self.event_bus = event_bus
        self.llm = llm_router
        self.memory: ContextMemory = None  # Injected by orchestrator
        self.logger = logging.getLogger(f"agent.{self.agent_name}")
    
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's primary task."""
        pass
    
    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> Any:
        if response_format:
            import json
            schema = response_format.model_json_schema()
            system_prompt += f"\n\nYou MUST return a JSON object that strictly conforms to the following JSON schema. Do not include any other text.\n{json.dumps(schema, indent=2)}\n"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return await self.llm.complete(messages, response_format)
    
    async def _publish_event(self, event_type: EventType, payload: Dict[str, Any]) -> None:
        event = Event(event_type=event_type, source=self.agent_name, payload=payload)
        await self.event_bus.publish(event)
    
    async def _subscribe_to_event(self, event_type: EventType) -> None:
        # In a real implementation we would bind a specific handler here
        pass
