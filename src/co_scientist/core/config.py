"""
Configuration management for Co-Scientist.
"""

from __future__ import annotations
import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
from pydantic import Field, field_validator, SecretStr, ValidationInfo, BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

class LLMConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env", extra="ignore")
    provider: Optional[str] = None
    model: Optional[str] = None
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_api_base: Optional[str] = None
    embedding_api_key: Optional[SecretStr] = None
    api_key: Optional[SecretStr] = None
    base_url: Optional[str] = None
    fallback_chain: List[str] = Field(default_factory=list)
    max_total_cost_usd: float = 50.0
    temperature: float = 0.7
    max_tokens: int = 4096

class SearchConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SEARCH_", extra="ignore")
    semantic_scholar_api_key: Optional[str] = None
    google_scholar_serpapi_key: Optional[str] = None
    serper_api_key: Optional[str] = None
    brave_api_key: Optional[str] = None
    tinyfish_api_key: Optional[str] = None

class AgentConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENTS_", extra="ignore")
    docker_memory_limit: str = "2g"
    docker_cpu_limit: float = 2.0
    docker_timeout_seconds: int = 300
    docker_image: str = "coscientist-sandbox:latest"
    debate_rounds: int = 3
    
    # Kimi Work swarm settings
    swarm_max_workers: int = 5          # cap parallel sub-agent dispatch
    swarm_ramp_per_cycle: int = 1       # add this many workers per cycle
    swarm_max_workers_ceiling: int = 8  # never exceed this

    # PlannerAgent settings
    planner_enabled: bool = True        # False = fall back to default plan without LLM planning call
    use_long_context: bool = True       # serialize full pool into PlannerAgent prompt
    long_context_top_n: int = 30        # how many hypotheses to include in planner context
    
    # LocalActionBridge / WebBridge
    browser_headless: bool = True
    browser_user_data_dir: Optional[str] = None   # path to Chrome profile for auth
    browser_timeout_ms: int = 30_000
    
    # Governance gate
    governance_auto_approve_threshold: float = 0.65
    governance_interactive: bool = False
    
    # Audit log
    audit_log_path: str = "./data/audit/audit.jsonl"

class WorkerConfig(BaseSettings):
    """JARVIS worker-loop settings (CLAUDE.md §6 `agent:` section)."""
    model_config = SettingsConfigDict(env_prefix="AGENT_", env_file=".env", extra="ignore")
    max_turns: int = 50              # hard cap on Think→Plan→Act→Observe→Verify iterations
    max_task_depth: int = 4          # DAG recursion depth cap
    context_compress_at: float = 0.92  # compress when this fraction of context is used
    budget_usd: float = 5.0          # per-task cost cap
    context_token_budget: int = 200_000  # assumed model context window for compression math

class SandboxConfig(BaseSettings):
    """Execution sandbox settings (CLAUDE.md §6 `sandbox:` section)."""
    model_config = SettingsConfigDict(env_prefix="SANDBOX_", env_file=".env", extra="ignore")
    provider: str = "e2b"            # "e2b" | "docker" | "local"; resolves with graceful fallback
    e2b_api_key: Optional[SecretStr] = None
    e2b_timeout_seconds: int = 300
    docker_memory_limit: str = "2g"
    docker_cpu_limit: float = 2.0
    docker_image: str = "coscientist-sandbox:latest"
    local_timeout_seconds: int = 120  # local fallback subprocess timeout

class ToolForgeConfig(BaseSettings):
    """Dynamic tool forge settings (CLAUDE.md §6 `tool_forge:` section)."""
    model_config = SettingsConfigDict(env_prefix="TOOL_FORGE_", env_file=".env", extra="ignore")
    max_repair_attempts: int = 5
    library_path: str = "data/tool_library"
    vector_search_top_k: int = 5

class SwarmConfig(BaseSettings):
    """Worker-swarm settings (CLAUDE.md §6 `swarm:` section)."""
    model_config = SettingsConfigDict(env_prefix="SWARM_", env_file=".env", extra="ignore")
    max_workers: int = 10
    planner_enabled: bool = True
    use_long_context: bool = True
    long_context_top_n: int = 30

class DatabaseConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DB_", extra="ignore")
    url: str = "sqlite+aiosqlite:///./co_scientist.db"
    vector_db_path: str = "./data/vector_db"

class ExperimentConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EXP_", extra="ignore")
    name: str = "generic_discovery"
    max_cycles: int = 5
    hypotheses_per_cycle: int = 10
    proximity_threshold: float = 0.85
    top_k_for_evolution: int = 3

    # Self-improvement loop
    improvement_loop_enabled: bool = True
    improvement_history_window: int = 3   # how many past cycles to aggregate

class DomainConfig(BaseSettings):
    """User-defined domain settings - loaded from YAML or env."""
    model_config = SettingsConfigDict(env_prefix="DOMAIN_", extra="ignore")
    name: str = "generic"
    safety_module: str = "default_safety"   # pluggable
    validation_module: str = "default_validator"

class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="COS_",
        extra="ignore",
        env_file=".env",
        env_nested_delimiter="__"
    )

    domain: DomainConfig = Field(default_factory=DomainConfig)
    debug: bool = False
    
    llm: LLMConfig = Field(default_factory=LLMConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    agents: AgentConfig = Field(default_factory=AgentConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)

    # JARVIS migration sections (CLAUDE.md §6)
    agent: WorkerConfig = Field(default_factory=WorkerConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    tool_forge: ToolForgeConfig = Field(default_factory=ToolForgeConfig)
    swarm: SwarmConfig = Field(default_factory=SwarmConfig)

    def get_candidate_model_class(self) -> Type[BaseModel]:
        from co_scientist.core.domain_models import Candidate
        return Candidate

def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or environment."""
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            yaml_data = yaml.safe_load(f) or {}
        return Config(**yaml_data)
    return Config()
