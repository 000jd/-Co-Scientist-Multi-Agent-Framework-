"""Tool ABC and shared types (CLAUDE.md §4: tool logic lives in the Tool ABC,
never inline in a worker's ``execute``)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ToolResult(BaseModel):
    """Uniform result of any tool invocation."""

    ok: bool
    output: str = ""
    error: Optional[str] = None
    artifacts: List[str] = []
    metadata: Dict[str, Any] = {}


@dataclass
class ToolContext:
    """Per-task execution context handed to every tool.

    ``workdir`` is the project-scoped sandbox root — tools must not touch the host
    filesystem outside it (CLAUDE.md §4 "Must not do").
    """

    workdir: Path
    sandbox: Any  # ExecutionSandbox
    project_id: str = "default"
    guardian: Any = None  # GuardianAgent — gates every outbound action (§4)
    audit: Any = None  # AuditLogger
    extra: Dict[str, Any] = field(default_factory=dict)

    def resolve(self, rel_path: str) -> Path:
        """Resolve ``rel_path`` inside the workspace, refusing escapes (path traversal)."""
        base = self.workdir.resolve()
        target = (base / rel_path).resolve()
        if base != target and base not in target.parents:
            raise ValueError(f"path escapes project workspace: {rel_path}")
        return target


class Tool(ABC):
    """Abstract base for every worker tool."""

    name: str = "tool"
    description: str = ""
    input_schema: Dict[str, Any] = {}

    @abstractmethod
    async def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        """Execute the tool with validated ``args`` in the given context."""
        raise NotImplementedError

    def spec(self) -> Dict[str, Any]:
        """JSON-serialisable spec for inclusion in a planner prompt."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
