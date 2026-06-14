"""Persistent library of forged tools (CLAUDE.md §1.3, §7 Phase 4).

Each tool the DynamicToolForge builds is stored as ``<library_path>/<name>.json`` and
looked up later by semantic/keyword search so it is forged once and reused thereafter.

Search prefers ChromaDB (lazy-imported) when it is installed; otherwise it degrades
gracefully to a deterministic ``difflib``/keyword similarity over name + description so
the library works with zero extra dependencies.
"""

from __future__ import annotations

import difflib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from pydantic import BaseModel, Field

log = structlog.get_logger("agent.tools.tool_library")

_WORD_RE = re.compile(r"[a-z0-9]+")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ToolSpec(BaseModel):
    """A self-contained, re-runnable script tool."""

    name: str
    description: str
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    script: str
    language: str = "python"
    # Shell command template, e.g. ``python3 {script_path} {args}``.
    entrypoint: str = "python3 {script_path} {args}"
    created_at: str = Field(default_factory=_now_iso)
    sample_input: Dict[str, Any] = Field(default_factory=dict)
    sample_output: str = ""


def _keywords(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


class ToolLibrary:
    """Stores ToolSpecs on disk and finds the best match for a task description."""

    def __init__(self, library_path: str = "data/tool_library"):
        self.path = Path(library_path)
        self.path.mkdir(parents=True, exist_ok=True)

    # ── persistence ─────────────────────────────────────────────────────────
    def _file(self, name: str) -> Path:
        return self.path / f"{name}.json"

    def save(self, spec: ToolSpec) -> None:
        self._file(spec.name).write_text(
            spec.model_dump_json(indent=2), encoding="utf-8"
        )
        log.info("tool_library.saved", name=spec.name, path=str(self._file(spec.name)))

    def load(self, name: str) -> Optional[ToolSpec]:
        f = self._file(name)
        if not f.exists():
            return None
        try:
            return ToolSpec.model_validate_json(f.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            log.warning("tool_library.load_failed", name=name, error=str(e))
            return None

    def all(self) -> List[ToolSpec]:
        specs: List[ToolSpec] = []
        for f in sorted(self.path.glob("*.json")):
            try:
                specs.append(
                    ToolSpec.model_validate_json(f.read_text(encoding="utf-8"))
                )
            except Exception as e:  # noqa: BLE001
                log.warning("tool_library.load_failed", path=str(f), error=str(e))
        return specs

    # ── search ──────────────────────────────────────────────────────────────
    def search(self, query: str, top_k: int = 5) -> List[ToolSpec]:
        """Return up to ``top_k`` specs most relevant to ``query`` (best first)."""
        specs = self.all()
        if not specs:
            return []
        ranked = self._search_chroma(query, specs, top_k)
        if ranked is not None:
            return ranked
        return self._search_keyword(query, specs, top_k)

    def _search_keyword(
        self, query: str, specs: List[ToolSpec], top_k: int
    ) -> List[ToolSpec]:
        q_words = set(_keywords(query))
        scored: List[tuple[float, ToolSpec]] = []
        for spec in specs:
            corpus = f"{spec.name} {spec.description}"
            doc_words = set(_keywords(corpus))
            overlap = len(q_words & doc_words)
            ratio = difflib.SequenceMatcher(None, query.lower(), corpus.lower()).ratio()
            # Keyword overlap dominates; fuzzy ratio breaks ties / catches morphology.
            score = overlap + ratio
            scored.append((score, spec))
        scored.sort(key=lambda s: s[0], reverse=True)
        return [spec for score, spec in scored[:top_k] if score > 0]

    def _search_chroma(
        self, query: str, specs: List[ToolSpec], top_k: int
    ) -> Optional[List[ToolSpec]]:
        """ChromaDB-backed semantic search; returns None when chromadb is unavailable."""
        try:
            import chromadb  # type: ignore
        except Exception:
            return None
        try:
            client = chromadb.EphemeralClient()
            coll = client.create_collection("tool_library")
            coll.add(
                ids=[s.name for s in specs],
                documents=[f"{s.name}. {s.description}" for s in specs],
            )
            res = coll.query(query_texts=[query], n_results=min(top_k, len(specs)))
            ids = (res.get("ids") or [[]])[0]
            by_name = {s.name: s for s in specs}
            return [by_name[i] for i in ids if i in by_name]
        except Exception as e:  # noqa: BLE001
            log.warning("tool_library.chroma_failed_fallback", error=str(e))
            return None
