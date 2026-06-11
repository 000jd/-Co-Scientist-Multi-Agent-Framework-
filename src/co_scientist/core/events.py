"""
Async Event Bus for inter-agent communication.
"""

import asyncio
from enum import Enum
from typing import Dict, Any, Callable, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone

def _now():
    return datetime.now(timezone.utc)

class EventType(str, Enum):
    # Pipeline lifecycle
    PIPELINE_STARTED = "pipeline_started"
    PIPELINE_COMPLETED = "pipeline_completed"
    CYCLE_STARTED = "cycle_started"
    CYCLE_COMPLETED = "cycle_completed"
    STAGE_STARTED = "stage_started"
    STAGE_COMPLETED = "stage_completed"
    
    # Hypothesis lifecycle
    HYPOTHESIS_GENERATED = "hypothesis_generated"
    HYPOTHESIS_REFLECTED = "hypothesis_reflected"
    HYPOTHESIS_CLUSTERED = "hypothesis_clustered"
    HYPOTHESIS_EVOLVED = "hypothesis_evolved"
    HYPOTHESIS_RANKED = "hypothesis_ranked"
    HYPOTHESIS_META_REVIEWED = "hypothesis_meta_reviewed"
    
    # Agent events
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"
    
    # Robin subsystem
    ROBIN_LITERATURE_FOUND = "robin_literature_found"
    ROBIN_ANALYSIS_COMPLETE = "robin_analysis_complete"
    ROBIN_CONSENSUS_REACHED = "robin_consensus_reached"
    ROBIN_DISAGREEMENT = "robin_disagreement"
    
    # Data events (Fix #1)
    DATA_FETCH_SUBMITTED = "data_fetch_submitted"
    DATA_FETCH_COMPLETED = "data_fetch_completed"
    DATA_FETCH_FAILED = "data_fetch_failed"
    
    # Safety events
    SAFETY_FLAG_RAISED = "safety_flag_raised"
    SAFETY_VERDICT_ISSUED = "safety_verdict_issued"
    GOVERNANCE_GATE_REQUIRED = "governance_gate_required"

class Event(BaseModel):
    event_type: EventType
    source: str
    payload: Dict[str, Any]
    timestamp: datetime = Field(default_factory=_now)
    correlation_id: Optional[str] = None

class EventBus:
    """Async event bus using asyncio.Queue and subscriber pattern."""
    
    def __init__(self):
        self.subscribers: Dict[EventType, List[Callable]] = {}
        self.queue = asyncio.Queue()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
    async def publish(self, event: Event) -> None:
        await self.queue.put(event)
        
    async def subscribe(self, event_type: EventType, handler: Callable) -> None:
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        if handler not in self.subscribers[event_type]:
            self.subscribers[event_type].append(handler)
            
    async def unsubscribe(self, event_type: EventType, handler: Callable) -> None:
        if event_type in self.subscribers and handler in self.subscribers[event_type]:
            self.subscribers[event_type].remove(handler)
            
    async def _process_events(self) -> None:
        while self._running:
            try:
                event = await self.queue.get()
                handlers = self.subscribers.get(event.event_type, [])
                for handler in handlers:
                    if asyncio.iscoroutinefunction(handler):
                        task = asyncio.create_task(handler(event))
                        def _handle_exception(t, evt=event):
                            try:
                                t.result()
                            except Exception as e:
                                print(f"Error in async event handler for {evt.event_type}: {e}")
                        task.add_done_callback(_handle_exception)
                    else:
                        try:
                            handler(event)
                        except Exception as e:
                            print(f"Error in sync event handler for {event.event_type}: {e}")
                self.queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error processing event loop: {e}")
                
    async def start(self) -> None:
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._process_events())
            
    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
