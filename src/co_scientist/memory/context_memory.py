"""
Context Memory with ChromaDB.
"""

from typing import List, Dict, Any

class ContextMemory:
    """ChromaDB-backed vector store for hypothesis embeddings."""
    
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = persist_dir
        try:
            import chromadb
            self.client = chromadb.PersistentClient(path=persist_dir)
            self.collection = self.client.get_or_create_collection(
                name="hypotheses",
                metadata={"hnsw:space": "cosine"}
            )
        except ImportError:
            self.client = None
            self.collection = None
            
    async def store_embedding(self, hypothesis_id: str, embedding: List[float], metadata: Dict[str, Any]) -> None:
        if self.collection:
            import asyncio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: self.collection.add(
                ids=[hypothesis_id],
                embeddings=[embedding],
                metadatas=[metadata]
            ))
            
    async def query_similar(self, embedding: List[float], top_k: int = 10) -> List[Dict]:
        if not self.collection:
            return []
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )
        return results
        
    async def delete_embedding(self, hypothesis_id: str) -> None:
        if self.collection:
            self.collection.delete(ids=[hypothesis_id])
