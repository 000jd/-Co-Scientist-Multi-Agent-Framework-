"""
Multi-provider LLM router with fallback chain.
"""

from typing import List, Dict, Any, Optional, Type, AsyncIterator
from pydantic import BaseModel
import json
import logging
import asyncio
from co_scientist.core.config import LLMConfig

# Mock classes for providers since this is a complex integration.
class OpenAIProvider:
    def __init__(self, api_key: str, model: str, embedding_model: str, cost_tracker: Any, base_url: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        self.embedding_model = embedding_model
        self.cost_tracker = cost_tracker
        try:
            import openai
            kwargs = {"api_key": api_key, "timeout": 600.0}
            if base_url:
                kwargs["base_url"] = base_url
            self.client = openai.AsyncClient(**kwargs)
        except ImportError:
            self.client = None
            
    @staticmethod
    def _strip_json_fence(text: str) -> str:
        """Strip ```json ... ``` fences that models sometimes emit."""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    async def complete(self, messages: List[Dict[str, str]], response_format: Optional[Type[BaseModel]] = None) -> Any:
        if not self.client:
            raise RuntimeError("OpenAI client not installed")
        if response_format:
            # Pydantic parsing mock for simplicity
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"} # Simplified
            )
            if hasattr(response, 'usage') and response.usage:
                await self.cost_tracker.add_cost(response.usage.prompt_tokens, response.usage.completion_tokens, self.model)
            content = response.choices[0].message.content
            content = self._strip_json_fence(content)
            return response_format.model_validate_json(content)
        else:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            if hasattr(response, 'usage') and response.usage:
                await self.cost_tracker.add_cost(response.usage.prompt_tokens, response.usage.completion_tokens, self.model)
            return response.choices[0].message.content

    async def embed(self, text: str) -> List[float]:
        if not self.client:
            raise RuntimeError("OpenAI client not installed")
        response = await self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        if hasattr(response, 'usage') and response.usage:
            await self.cost_tracker.add_cost(response.usage.prompt_tokens, 0, self.embedding_model)
        return response.data[0].embedding

class OllamaEmbeddingProvider:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = None

    async def embed(self, text: str) -> List[float]:
        import httpx
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
            )
            resp.raise_for_status()
            return resp.json()["embedding"]


class AnthropicProvider:
    """Native Anthropic SDK provider (claude-* models)."""

    def __init__(self, api_key: str, model: str, cost_tracker: Any):
        self.api_key = api_key
        self.model = model
        self.cost_tracker = cost_tracker
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
        except ImportError:
            self.client = None

    @staticmethod
    def _strip_json_fence(text: str) -> str:
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    async def complete(self, messages: List[Dict[str, str]], response_format=None) -> Any:
        if not self.client:
            raise RuntimeError("anthropic package not installed — run: pip install anthropic")
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msgs = [m for m in messages if m["role"] != "system"]
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=user_msgs,
        )
        if hasattr(response, "usage") and response.usage:
            await self.cost_tracker.add_cost(
                response.usage.input_tokens,
                response.usage.output_tokens,
                self.model,
            )
        text = response.content[0].text
        if response_format:
            clean = self._strip_json_fence(text)
            return response_format.model_validate_json(clean)
        return text

    async def embed(self, text: str) -> List[float]:
        raise NotImplementedError("Anthropic does not provide embeddings — configure a separate embedding_provider")

class SentenceTransformerEmbeddingProvider:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    async def embed(self, text: str) -> List[float]:
        import asyncio
        model = self._get_model()
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, model.encode, text)
        return embedding.tolist()

class CostTracker:
    def __init__(self, max_total: float):
        import asyncio
        self.max_total = max_total
        self.current_total = 0.0
        self._lock = asyncio.Lock()
        
    async def add_cost(self, input_tokens: int, output_tokens: int, model: str = "gpt-4o"):
        import asyncio
            
        # GPT-4o pricing: $5/1M input, $15/1M output
        rates = {
            "gpt-4o": (5.0, 15.0),
            "gpt-4o-mini": (0.15, 0.60),
            "claude-3-5-sonnet": (3.0, 15.0),
            "text-embedding-3-small": (0.02, 0.0),
            "moonshotai/kimi-k2.6:free": (0.0, 0.0),
        }
        in_rate, out_rate = rates.get(model, (5.0, 15.0))
        cost = (input_tokens / 1_000_000 * in_rate) + (output_tokens / 1_000_000 * out_rate)
        
        async with self._lock:
            self.current_total += cost
            if self.current_total > self.max_total:
                raise RuntimeError(f"BUDGET EXCEEDED: ${self.current_total:.2f} / ${self.max_total}")
            return self.current_total

class LLMRouter:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.fallback_chain = config.fallback_chain
        self.cost_tracker = CostTracker(max_total=config.max_total_cost_usd)
        self.logger = logging.getLogger("infrastructure.llm")
        self.chat_providers = {}
        self.embedding_providers = {}
        # Kimi Work: semaphore caps parallel K2.6 calls (free tier = 3 concurrent max)
        self._semaphore = asyncio.Semaphore(3)
        
        api_key = config.api_key.get_secret_value() if config.api_key else ""
        if "openai" in self.fallback_chain:
            self.chat_providers["openai"] = OpenAIProvider(api_key, config.model, config.embedding_model, self.cost_tracker)
        if "openrouter" in self.fallback_chain:
            base_url = getattr(config, 'base_url', "https://openrouter.ai/api/v1")
            self.chat_providers["openrouter"] = OpenAIProvider(
                api_key, config.model, config.embedding_model, self.cost_tracker, base_url=base_url
            )
        if "ollama" in self.fallback_chain:
            self.chat_providers["ollama"] = OpenAIProvider(
                "ollama", config.model, config.embedding_model, self.cost_tracker, base_url="http://localhost:11434/v1"
            )
        if "anthropic" in self.fallback_chain:
            anthropic_key = api_key  # reuse LLM_API_KEY or override via ANTHROPIC_API_KEY env var
            import os
            anthropic_key = os.environ.get("ANTHROPIC_API_KEY", anthropic_key)
            anthropic_model = getattr(config, "anthropic_model", "claude-3-5-sonnet-20241022")
            self.chat_providers["anthropic"] = AnthropicProvider(anthropic_key, anthropic_model, self.cost_tracker)
            
        # Initialize embedding provider
        emb_provider = getattr(config, "embedding_provider", "openai")
        emb_model = getattr(config, "embedding_model", "text-embedding-3-small")
        emb_api_key = config.embedding_api_key.get_secret_value() if config.embedding_api_key else api_key
        emb_api_base = getattr(config, "embedding_api_base", None)
        
        if emb_provider == "openai":
            self.embedding_providers["default"] = OpenAIProvider(
                emb_api_key, config.model, emb_model, self.cost_tracker, base_url=emb_api_base
            )
        elif emb_provider == "ollama":
            self.embedding_providers["default"] = OllamaEmbeddingProvider(
                base_url=emb_api_base or "http://localhost:11434",
                model=emb_model,
            )
        elif emb_provider == "sentence_transformers":
            self.embedding_providers["default"] = SentenceTransformerEmbeddingProvider(
                model_name=emb_model,
            )
            
    async def complete(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Type[BaseModel]] = None,
        provider: Optional[str] = None,
    ) -> Any:
        async with self._semaphore:   # rate-gate for swarm parallelism
            providers_to_try = [provider] if provider else self.fallback_chain
            last_error = None
            
            for p in providers_to_try:
                if p not in self.chat_providers:
                    continue
                for attempt in range(4):
                    try:
                        result = await self.chat_providers[p].complete(messages, response_format)
                        return result
                    except Exception as e:
                        last_error = e
                        if "429" in str(e) and attempt < 3:
                            sleep_time = 5 * (2 ** attempt)  # exponential backoff
                            self.logger.info(f"Rate limited. Waiting {sleep_time}s...")
                            await asyncio.sleep(sleep_time)
                            continue
                        break
                        
            raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")

    async def embed(self, text: str, provider: Optional[str] = None) -> List[float]:
        provider_name = provider or "default"
        if provider_name not in self.embedding_providers:
            raise RuntimeError(f"Embedding provider '{provider_name}' not configured")
        try:
            return await self.embedding_providers[provider_name].embed(text)
        except Exception as e:
            self.logger.warning(f"Embedding failed: {e}")
            if "sentence_transformers" not in self.embedding_providers:
                raise
            return await self.embedding_providers["sentence_transformers"].embed(text)
