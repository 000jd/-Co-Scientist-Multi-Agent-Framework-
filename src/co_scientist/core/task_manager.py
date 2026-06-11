"""Daemon-style task manager for unlimited convergence-based execution."""

import asyncio
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from co_scientist.core.orchestrator import CoScientistOrchestrator
from co_scientist.core.config import Config


class ConvergenceTracker:
    """Detects when the pipeline has stopped producing better ideas."""
    
    def __init__(self, window_size: int = 3, improvement_threshold: float = 0.5):
        self.window_size = window_size
        self.improvement_threshold = improvement_threshold
        self.leaderboard_history: list = []
        self.best_scores: list = []
        
    def record(self, leaderboard_display: list) -> None:
        """Call this every cycle."""
        self.leaderboard_history.append(leaderboard_display)
        if leaderboard_display:
            best = max(entry.get("adjusted_score", 0) for entry in leaderboard_display)
            self.best_scores.append(best)
        else:
            self.best_scores.append(0.0)
            
    @property
    def is_converged(self) -> bool:
        """True if leaderboard hasn't improved in N cycles."""
        if len(self.best_scores) < self.window_size + 1:
            return False
            
        recent_best = max(self.best_scores[-self.window_size:])
        previous_best = max(self.best_scores[:-(self.window_size)])
        
        # Also check if top-3 is stable (same IDs)
        if len(self.leaderboard_history) >= self.window_size:
            recent_ids = [
                [e["id"] for e in lb[:3]] 
                for lb in self.leaderboard_history[-self.window_size:]
            ]
            # If all recent top-3 are identical, we're stable
            if all(ids == recent_ids[0] for ids in recent_ids):
                return True
                
        return (recent_best - previous_best) < self.improvement_threshold


class SelfImprovementLoop:
    def __init__(self, llm_router):
        self.llm_router = llm_router
        self.history = []
        
    async def extract_structured_feedback(self, raw_insights: str, leaderboard: list) -> dict:
        prompt = f"""Extract actionable structured feedback for the next generation cycle.
        
Raw Insights: {raw_insights}

Focus on what worked, what failed, and specific parameters to tune.
Return JSON with 'what_worked' (list), 'what_failed' (list), and 'recommended_focus' (str)."""
        
        try:
            from pydantic import BaseModel
            class FeedbackForm(BaseModel):
                what_worked: list[str]
                what_failed: list[str]
                recommended_focus: str
                
            res = await self.llm_router.complete(
                messages=[{"role": "user", "content": prompt}],
                response_format=FeedbackForm
            )
            feedback = res.model_dump()
        except Exception:
            feedback = {"what_worked": [], "what_failed": [], "recommended_focus": raw_insights}
            
        self.history.append({"feedback": feedback, "leaderboard_top": leaderboard[:3] if leaderboard else []})
        return feedback
        
    def build_generation_context(self) -> str:
        if not self.history:
            return ""
        latest = self.history[-1]["feedback"]
        return f"Recommended focus: {latest['recommended_focus']}\nAvoid: {', '.join(latest['what_failed'])}"



class TaskManager:
    """Runs the orchestrator until convergence or manual stop."""
    
    def __init__(
        self,
        config: Config,
        checkpoint_dir: str = "./data/checkpoints",
        max_cycles: Optional[int] = None,  # None = infinite
        convergence_window: int = 3,
    ):
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_cycles = max_cycles
        self.convergence = ConvergenceTracker(window_size=convergence_window)
        self.orchestrator: Optional[CoScientistOrchestrator] = None
        self._stop_requested = False
        
    async def run(self, research_goal: str) -> Dict[str, Any]:
        """Run until convergence, crash, or manual stop."""
        candidate_model = self.config.get_candidate_model_class()
        self.orchestrator = CoScientistOrchestrator(self.config, candidate_model)
        
        await self.orchestrator.initialize()
        self.improvement_loop = SelfImprovementLoop(self.orchestrator.llm_router)
        
        cycle = 0
        previous_insights = ""
        
        while not self._stop_requested:
            cycle += 1
            
            # Hard limit override (if you want a safety cap)
            if self.max_cycles and cycle > self.max_cycles:
                print(f"[TaskManager] Hard limit reached: {self.max_cycles}")
                break
                
            print(f"\n{'='*50}")
            print(f"[TaskManager] Starting Cycle {cycle} | Goal: {research_goal[:50]}...")
            print(f"{'='*50}")
            
            # Run one cycle manually (bypassing the capped run_pipeline)
            await self.orchestrator._run_cycle(cycle, research_goal, previous_insights)
            
            # Record state for convergence
            lb = self.orchestrator.leaderboard.to_display()
            self.convergence.record(lb)
            
            # Save checkpoint every cycle
            await self._checkpoint(cycle)
            
            # Supervisor convergence check (enhanced)
            supervisor_res = await self.orchestrator.supervisor_agent.execute({
                "hypotheses": list(self.orchestrator.pool.hypotheses.values()),
                "cycle": cycle,
                "convergence_tracker": self.convergence,  # Pass it in
            })
            
            if supervisor_res.get("action") == "trigger_meta_review":
                print("[TaskManager] Supervisor triggered meta-review (convergence detected).")
                break
                
            if self.convergence.is_converged:
                print(f"[TaskManager] Convergence detected after {cycle} cycles. Top score stable.")
                break
                
            # Generate insights for next cycle
            mr_res = await self.orchestrator.meta_review_agent.execute({
                "leaderboard": lb,
                "hypotheses": list(self.orchestrator.pool.hypotheses.values()),
                "cycle": cycle
            })
            raw_insights = mr_res.get("report", "")
            feedback = await self.improvement_loop.extract_structured_feedback(
                raw_insights,
                self.orchestrator.leaderboard.to_display()
            )
            previous_insights = self.improvement_loop.build_generation_context()
            print(f"[TaskManager] Meta-review insight: {raw_insights[:200]}...")
            
        # Final meta-review
        final_lb = self.orchestrator.leaderboard.to_display()
        await self.orchestrator.meta_review_agent.execute({
            "leaderboard": final_lb,
            "hypotheses": list(self.orchestrator.pool.hypotheses.values()),
            "cycle": "Final"
        })
        
        await self._checkpoint(cycle, final=True)
        await self.orchestrator.shutdown()
        
        return {
            "cycles_run": cycle,
            "final_leaderboard": final_lb,
            "total_hypotheses": len(self.orchestrator.pool),
            "converged": self.convergence.is_converged,
        }
        
    async def _checkpoint(self, cycle: int, final: bool = False) -> None:
        """Save state to disk."""
        suffix = "final" if final else f"cycle_{cycle}"
        path = self.checkpoint_dir / f"checkpoint_{suffix}.json"
        
        data = {
            "cycle": cycle,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "leaderboard": self.orchestrator.leaderboard.to_display(),
            "hypotheses": [h.model_dump(mode='json') for h in self.orchestrator.pool.hypotheses.values()],
            "convergence_scores": self.convergence.best_scores,
            "improvement_history": self.improvement_loop.history,
        }
        path.write_text(json.dumps(data, indent=2, default=str))
        print(f"[TaskManager] Checkpoint saved: {path}")
        
    def resume_from_checkpoint(self, checkpoint_file: str) -> None:
        """Resume state from a checkpoint file (stub)."""
        pass
        
    def stop(self) -> None:
        """Call this from a signal handler or another thread."""
        self._stop_requested = True
