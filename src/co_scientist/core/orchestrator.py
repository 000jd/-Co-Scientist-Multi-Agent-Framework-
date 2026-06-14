"""
Co-Scientist Pipeline Orchestrator.
"""

import asyncio
from typing import Type, TypeVar, Optional, List, Dict, Any
import importlib
from pydantic import BaseModel
import logging

from co_scientist.core.config import Config
from co_scientist.core.events import EventBus, EventType, Event
from co_scientist.core.database import DatabaseManager
from co_scientist.core.hypothesis import HypothesisPool, CandidateT, Hypothesis
from co_scientist.memory.context_memory import ContextMemory
from co_scientist.ranking.trueskill_adapter import CombinedLeaderboard
from co_scientist.infrastructure.llm import LLMRouter

# Import agents
from co_scientist.agents.co_scientist.generation import GenerationAgent
from co_scientist.agents.co_scientist.reflection import ReflectionAgent
from co_scientist.agents.co_scientist.proximity import ProximityAgent
from co_scientist.agents.co_scientist.evolution import EvolutionAgent
from co_scientist.agents.co_scientist.ranking import RankingAgent
from co_scientist.agents.co_scientist.supervisor import SupervisorAgent
from co_scientist.agents.co_scientist.meta_review import MetaReviewAgent
from co_scientist.agents.co_scientist.agentic_search import AgenticSearchAgent

from co_scientist.tools.registry import ToolRegistry
from co_scientist.validation.literature import LiteratureValidator
from co_scientist.validation.sandbox import SimulationValidator, ConsensusValidator
from co_scientist.agents.swarm_orchestrator import SwarmOrchestrator
from co_scientist.infrastructure.audit_logger import AuditLogger
from co_scientist.safety.prompt_injection_guard import PromptInjectionGuard
from co_scientist.safety.governance_gate import GovernanceGate
from co_scientist.infrastructure.local_action_bridge import LocalActionBridge

try:
    import chromadb
except ImportError:
    chromadb = None

try:
    import hdbscan
except ImportError:
    hdbscan = None

class CoScientistOrchestrator:
    """Main pipeline orchestrator, parameterized by domain model class."""
    
    def __init__(
        self,
        config: Config,
        candidate_model: Type[CandidateT],
    ):
        self.config = config
        self.candidate_model = candidate_model
        self.logger = logging.getLogger("orchestrator")
        
        self.pool = HypothesisPool[candidate_model](
            candidate_model=candidate_model.__name__
        )
        self.memory = ContextMemory(config.database.vector_db_path)
        self.event_bus = EventBus()
        self.database = DatabaseManager(config.database.url)
        
        self.llm_router = LLMRouter(config.llm)
        self.leaderboard = CombinedLeaderboard()
        
        self.audit_logger = AuditLogger(getattr(config.agents, "audit_log_path", "./data/audit/audit.jsonl"))
        self.injection_guard = PromptInjectionGuard()
        
        self.browser_bridge = LocalActionBridge(
            headless=getattr(config.agents, "browser_headless", True),
            timeout_ms=getattr(config.agents, "browser_timeout_ms", 30000)
        )
        self.tool_registry = ToolRegistry(audit_logger=self.audit_logger, browser_bridge=self.browser_bridge)
        self.safety_checker = self._load_safety_checker(config.domain.safety_module)
        
        # Initialize agents
        self.generation_agent = GenerationAgent(self.config, self.event_bus, self.llm_router, self.tool_registry)
        self.reflection_agent = ReflectionAgent(self.config, self.event_bus, self.llm_router, self.safety_checker)
        self.proximity_agent = ProximityAgent(self.config, self.event_bus, self.llm_router)
        self.evolution_agent = EvolutionAgent(self.config, self.event_bus, self.llm_router)
        self.ranking_agent = RankingAgent(self.config, self.event_bus, self.llm_router, self.leaderboard)
        self.supervisor_agent = SupervisorAgent(self.config, self.event_bus, self.llm_router, self.tool_registry)
        self.meta_review_agent = MetaReviewAgent(self.config, self.event_bus, self.llm_router)
        self.agentic_search = AgenticSearchAgent(self.config, self.event_bus, self.llm_router, self.tool_registry, self.injection_guard)
        
        self.governance_gate = GovernanceGate(
            auto_approve_threshold=getattr(config.agents, "governance_auto_approve_threshold", 0.65),
            interactive=getattr(config.agents, "governance_interactive", False),
            audit_logger=self.audit_logger
        )
        
        self.swarm_orchestrator = SwarmOrchestrator(
            self.config, self.event_bus, self.llm_router,
            sub_agents={
                "generation": self.generation_agent,
                "search": self.agentic_search,
                "reflection": self.reflection_agent,
                "evolution": self.evolution_agent,
            },
            governance_gate=self.governance_gate,
            audit_logger=self.audit_logger,
            injection_guard=self.injection_guard
        )
        
        # Validation subsystem
        self.literature_validator = LiteratureValidator(self.config, self.event_bus, self.llm_router)
        self.simulation_validator = SimulationValidator(self.config, self.event_bus, self.llm_router)
        self.consensus_validator = ConsensusValidator(self.config, self.event_bus, self.llm_router)
        
        # Inject shared memory into all agents
        for agent in [
            self.generation_agent, self.reflection_agent, self.proximity_agent,
            self.evolution_agent, self.ranking_agent, self.supervisor_agent,
            self.meta_review_agent, self.agentic_search, self.swarm_orchestrator,
            self.literature_validator, self.simulation_validator, self.consensus_validator
        ]:
            agent.memory = self.memory
            
    def _load_safety_checker(self, module_path: str):
        if module_path == "default_safety":
            from co_scientist.safety.base import DefaultSafetyChecker
            return DefaultSafetyChecker()
        parts = module_path.split('.')
        module_name = '.'.join(parts[:-1])
        class_name = parts[-1]
        try:
            module = importlib.import_module(module_name)
            checker_class = getattr(module, class_name)
            return checker_class()
        except Exception as e:
            from co_scientist.safety.base import DefaultSafetyChecker
            return DefaultSafetyChecker()
        
    async def initialize(self):
        await self.database.init_db()
        await self.event_bus.start()
        
        await self.browser_bridge.start()
        
        # Note: EventType.HYPOTHESIS_GENERATED subscription removed to prevent double-adds.
        
    async def _on_hypothesis_created(self, event):
        # Now obsolete, kept for compatibility if needed elsewhere
        pass
        
    async def shutdown(self):
        await self.browser_bridge.stop()
        await self.event_bus.stop()
        
    async def run_pipeline(
        self,
        research_goal: str,
        max_cycles: Optional[int] = None,
    ) -> HypothesisPool[CandidateT]:
        
        await self.initialize()
        await self.event_bus.publish(Event(event_type=EventType.PIPELINE_STARTED, source="orchestrator", payload={"goal": research_goal}))
        
        cycles = max_cycles or self.config.experiment.max_cycles
        previous_insights = ""
        
        for cycle in range(1, cycles + 1):
            await self._run_cycle(cycle, research_goal, previous_insights)
            
            # Supervisor check
            supervisor_res = await self.supervisor_agent.execute({
                "hypotheses": list(self.pool.hypotheses.values()),
                "cycle": cycle
            })
            
            if supervisor_res.get("action") == "trigger_meta_review":
                break
                
            # Generate insights for the next cycle
            if cycle < cycles:
                mr_res = await self.meta_review_agent.execute({
                    "leaderboard": self.leaderboard.to_display(),
                    "hypotheses": list(self.pool.hypotheses.values()),
                    "cycle": cycle
                })
                previous_insights = mr_res.get("report", "")
                
        # Final Meta-review
        await self.meta_review_agent.execute({
            "leaderboard": self.leaderboard.to_display(),
            "hypotheses": list(self.pool.hypotheses.values()),
            "cycle": "Final"
        })
        
        await self.event_bus.publish(Event(event_type=EventType.PIPELINE_COMPLETED, source="orchestrator", payload={"cycles_run": cycles}))
        await self.shutdown()
        return self.pool
        
    async def _run_cycle(self, cycle_number: int, research_goal: str, previous_insights: str = "", convergence_info: str = "") -> None:
        # 1. Swarm Fan-out
        swarm_res = await self.swarm_orchestrator.execute({
            "research_goal": research_goal,
            "pool": self.pool,
            "cycle": cycle_number,
            "previous_insights": previous_insights,
            "max_workers": getattr(self.config.agents, "swarm_max_workers", 5),
            "convergence_info": convergence_info,
        })
        
        merged = swarm_res.get("merged")
        if merged:
            from co_scientist.core.hypothesis import Hypothesis
            from co_scientist.core.domain_models import Candidate
            
            for h in merged.new_hypotheses:
                if isinstance(h, Hypothesis):
                    self.pool.add(h)
                elif isinstance(h, dict) and "id" in h:
                    if "candidates" in h and isinstance(h["candidates"], list):
                        h["candidates"] = [
                            Candidate(**c) if isinstance(c, dict) else c 
                            for c in h["candidates"]
                        ]
                    try:
                        hyp = Hypothesis[self.candidate_model](**h)
                        self.pool.add(hyp)
                    except Exception as e:
                        self.logger.error(f"Failed to reconstruct hypothesis: {e}")
        
        # 2. Reflection (guaranteed)
        await self.reflection_agent.execute({
            "hypotheses": list(self.pool.hypotheses.values())
        })
        
        # 3. Proximity
        await self.proximity_agent.execute({
            "hypotheses": list(self.pool.hypotheses.values())
        })
                
        # 4. Evolution (top-K only)
        top_k = sorted(
            self.pool.hypotheses.values(), 
            key=lambda hyp: hyp.trueskill_rating.conservative_rating, 
            reverse=True
        )[:getattr(self.config.experiment, "top_k_for_evolution", 10)]
        
        if top_k:
            await self.evolution_agent.execute({
                "top_hypotheses": top_k
            })
            
        # 5. Ranking
        rank_res = await self.ranking_agent.execute({
            "hypotheses": list(self.pool.hypotheses.values())
        })
        
        if len(self.pool) > 200:
            # Prune worst hypotheses
            worst = sorted(self.pool.hypotheses.values(), 
                           key=lambda h: h.trueskill_rating.conservative_rating)[:50]
            for h in worst:
                del self.pool.hypotheses[h.id]
            print(f"[Orchestrator] Pruned 50 low-ranking hypotheses to prevent memory bloat.")
            
        # Real Database Persistence
        for h in self.pool.hypotheses.values():
            await self.database.save_hypothesis(h)
        
    def _get_candidate_model_class(self) -> Type[CandidateT]:
        return self.candidate_model
