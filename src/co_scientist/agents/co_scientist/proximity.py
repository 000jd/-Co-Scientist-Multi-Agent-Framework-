"""
Proximity Agent for Domain-Agnostic Interventions.
"""

from typing import Dict, Any, List
import numpy as np

from co_scientist.agents.base import BaseAgent
from co_scientist.core.hypothesis import Hypothesis, HypothesisStatus

class ProximityAgent(BaseAgent):
    """Detects redundant hypotheses and clusters similar ones."""
    agent_name = "proximity"
    
    def __init__(self, config, event_bus, llm_router):
        super().__init__(config, event_bus, llm_router)
        self._embedding_model = None
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        hypotheses: List[Hypothesis] = context["hypotheses"]
        
        # 1. Generate embeddings
        embeddings = []
        for h in hypotheses:
            text = f"{h.title} {h.summary} {h.rationale}"
            for c in h.candidates:
                text += f" {c.name} {c.description}"
                
            # Mock embedding generation
            embedding = await self._compute_embedding(text)
            h.proximity.embedding = embedding
            embeddings.append(embedding)
            
        # 2. Cluster with HDBSCAN (Mocked)
        cluster_ids = self._cluster_hypotheses(embeddings)
        
        # 3. Assign cluster IDs
        for h, cid in zip(hypotheses, cluster_ids):
            h.proximity.cluster_id = cid
            h.status = HypothesisStatus.CLUSTERED
            
        # 4. Flag near-duplicates
        duplicates = self._find_near_duplicates(hypotheses, self.config.experiment.proximity_threshold)
        
        return {
            "hypotheses": hypotheses,
            "duplicate_pairs": duplicates,
        }
        
    async def _compute_embedding(self, text: str) -> List[float]:
        try:
            return await self.llm.embed(text)
        except Exception as e:
            self.logger.warning(f"LLM embedding failed: {e}, using local fallback")
            if self._embedding_model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                    self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                except ImportError:
                    self.logger.error("sentence-transformers not installed. Fallback to random embedding.")
                    import random
                    return [random.random() for _ in range(384)]
            return self._embedding_model.encode(text).tolist()
        
    def _cluster_hypotheses(self, embeddings: List[List[float]]) -> List[int]:
        try:
            import hdbscan
            if len(embeddings) < 2:
                return [0] * len(embeddings)
            clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
            labels = clusterer.fit_predict(np.array(embeddings))
            return labels.tolist()
        except ImportError:
            # Fallback mock
            return [0] * len(embeddings)
            
    def _find_near_duplicates(self, hypotheses: List[Hypothesis], threshold: float) -> List[tuple]:
        duplicates = []
        for i, h1 in enumerate(hypotheses):
            for j, h2 in enumerate(hypotheses[i+1:]):
                sim = h1.proximity.cosine_similarity(h2.proximity)
                if sim >= threshold:
                    duplicates.append((h1.id, h2.id))
        return duplicates
