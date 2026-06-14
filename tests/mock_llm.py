"""Test doubles for the LLM layer.

The real interface is ``LLMRouter.complete(messages: List[Dict], response_format=None,
provider=None) -> str | BaseModel``. ``MockLLMRouter`` matches it exactly so worker /
agent tests drive the real loops deterministically with no API calls.

(Replaces the previous mock, which imported BaseLLMProvider/Message/LLMResponse —
symbols that never existed in ``co_scientist.infrastructure.llm``.)
"""

from __future__ import annotations

import json
from typing import Any, Callable, List, Optional, Sequence, Type, Union

from pydantic import BaseModel

Response = Union[str, BaseModel, dict]


class MockLLMRouter:
    """Returns scripted responses in order (or via a callable router)."""

    def __init__(
        self,
        responses: Optional[Sequence[Response]] = None,
        router: Optional[
            Callable[[List[dict], Optional[Type[BaseModel]]], Response]
        ] = None,
        default: Response = 'FINAL_ANSWER({"answer": ""})',
    ):
        self._responses: List[Response] = list(responses or [])
        self._router = router
        self._default = default
        self.calls: List[List[dict]] = []
        self.cost_tracker = None  # worker reads getattr(llm, "cost_tracker", None)

    async def complete(
        self, messages, response_format: Optional[Type[BaseModel]] = None, provider=None
    ) -> Any:
        self.calls.append(messages)
        if self._router is not None:
            resp = self._router(messages, response_format)
        elif self._responses:
            resp = self._responses.pop(0)
        else:
            resp = self._default
        return _coerce(resp, response_format)

    async def embed(self, text: str, provider=None) -> List[float]:
        return [0.0] * 16


def _coerce(resp: Response, response_format: Optional[Type[BaseModel]]):
    if response_format is not None:
        if isinstance(resp, response_format):
            return resp
        if isinstance(resp, BaseModel):
            return resp
        if isinstance(resp, dict):
            return response_format.model_validate(resp)
        if isinstance(resp, str):
            return response_format.model_validate_json(resp)
    if isinstance(resp, BaseModel):
        return resp.model_dump_json()
    if isinstance(resp, dict):
        return json.dumps(resp)
    return resp
