"""
ToolRegistry with built-in scientific discovery tools.

Kimi Work pattern: agents call tools by name through the registry.
The registry handles routing, audit logging, and governance gating.
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, Callable, Awaitable, Optional, List
import logging

ToolFunction = Callable[[Dict[str, Any]], Awaitable[Any]]

logger = logging.getLogger("tool_registry")


class ToolResult:
    def __init__(self, success: bool, data: Any, error: Optional[str] = None):
        self.success = success
        self.data = data
        self.error = error


class ToolRegistry:
    def __init__(self, audit_logger=None, browser_bridge=None):
        self._tools: Dict[str, ToolFunction] = {}
        self.audit_logger = audit_logger
        self.browser_bridge = browser_bridge
        self._register_builtin_tools()
        
    def register(self, name: str, func: ToolFunction):
        self._tools[name] = func
        
    async def call(self, name: str, **kwargs) -> ToolResult:
        if name not in self._tools:
            return ToolResult(success=False, data=None, error=f"Tool '{name}' not registered")
        try:
            if self.audit_logger:
                await self.audit_logger.record("tool_call", {"tool": name, "kwargs_keys": list(kwargs.keys())})
            result = await self._tools[name](kwargs)
            return ToolResult(success=True, data=result)
        except Exception as e:
            logger.warning(f"Tool '{name}' failed: {e}")
            return ToolResult(success=False, data=None, error=str(e))
            
    def _register_builtin_tools(self):
        """Register the built-in tool set."""
        
        SANDBOX_ROOT = Path("./data/workspace").resolve()
        SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)
        
        def _validate_path(path_str: str) -> Path:
            p = Path(path_str).resolve()
            if not str(p).startswith(str(SANDBOX_ROOT)):
                raise PermissionError(f"Access denied: path outside sandbox: {p}")
            return p
        
        async def file_read(kwargs: Dict) -> str:
            path = _validate_path(kwargs["path"])
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            return path.read_text(encoding="utf-8")[:50_000]
            
        async def file_write(kwargs: Dict) -> bool:
            path = _validate_path(kwargs["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(kwargs["content"], encoding="utf-8")
            return True
            
        async def browser_navigate(kwargs: Dict) -> str:
            if self.browser_bridge:
                result = await self.browser_bridge.navigate(kwargs["url"])
                return result.content
            raise RuntimeError("No browser bridge configured")
            
        async def literature_search(kwargs: Dict) -> List[Dict]:
            from co_scientist.search.literature_search import LiteratureSearcher
            searcher = LiteratureSearcher(config={})
            return await searcher.search(
                kwargs["query"],
                max_results=kwargs.get("max_results", 10),
                sources=kwargs.get("sources", ["semantic_scholar", "arxiv"]),
            )
            
        async def list_files(kwargs: Dict) -> List[str]:
            base = _validate_path(kwargs.get("path", "."))
            pattern = kwargs.get("pattern", "**/*")
            return [str(p) for p in base.glob(pattern) if p.is_file()][:100]
            
        self.register("file_read", file_read)
        self.register("file_write", file_write)
        self.register("browser_navigate", browser_navigate)
        self.register("literature_search", literature_search)
        self.register("list_files", list_files)
        
    @property
    def available_tools(self) -> List[str]:
        return list(self._tools.keys())
