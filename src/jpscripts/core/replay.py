"""Agent trace replay functionality.

Provides replay and diff capabilities for agent traces:
    - Trace step comparison
    - Response diffing
    - Replay simulation
"""

from __future__ import annotations

import difflib
import json
from collections.abc import AsyncIterator, Iterable, Sequence

from pydantic import BaseModel, ConfigDict

from jpscripts.agent import AgentTraceStep
from jpscripts.providers import (
    CompletionOptions,
    CompletionResponse,
    LLMProvider,
    ProviderType,
    StreamChunk,
)
from jpscripts.providers import (
    Message as ProviderMessage,
)


class ReplayDivergenceError(Exception):
    """Raised when replayed prompts diverge from the recorded trace."""

    def __init__(self, message: str, diff: str | None = None) -> None:
        super().__init__(message)
        self.diff = diff


class RecordedAgentResponse(BaseModel):
    """Generic wrapper for recorded agent responses."""

    payload: dict[str, object]

    model_config = ConfigDict(extra="forbid")


def _normalize_messages(messages: Iterable[ProviderMessage]) -> list[tuple[str, str]]:
    return [(msg.role, msg.content) for msg in messages]


def _normalize_history(history: Sequence[dict[str, str]]) -> list[tuple[str, str]]:
    return [(entry.get("role", "") or "", entry.get("content", "") or "") for entry in history]


def _diff_histories(expected: Sequence[tuple[str, str]], actual: Sequence[tuple[str, str]]) -> str:
    expected_lines = [f"{role}: {content}" for role, content in expected]
    actual_lines = [f"{role}: {content}" for role, content in actual]
    diff = difflib.unified_diff(
        expected_lines,
        actual_lines,
        fromfile="trace_history",
        tofile="replay_history",
        lineterm="",
    )
    return "\n".join(diff)


class ReplayProvider(LLMProvider):
    """Deterministic provider that replays recorded trace steps."""

    def __init__(self, steps: Sequence[AgentTraceStep]) -> None:
        self._steps = list(steps)
        self._cursor = 0

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.ANTHROPIC  # Replay uses Anthropic as default type

    @property
    def default_model(self) -> str:
        return "replay"

    @property
    def available_models(self) -> tuple[str, ...]:
        return ("replay",)

    async def complete(
        self,
        messages: list[ProviderMessage],
        model: str | None = None,
        options: CompletionOptions | None = None,
    ) -> CompletionResponse:
        _ = model, options
        if self._cursor >= len(self._steps):
            raise ReplayDivergenceError("Replay exceeded recorded steps.")

        expected_step = self._steps[self._cursor]
        expected_history = _normalize_history(expected_step.input_history)
        incoming_history = _normalize_messages(messages)

        if incoming_history != expected_history:
            diff = _diff_histories(expected_history, incoming_history)
            raise ReplayDivergenceError("Replay diverged from recorded history.", diff=diff)

        self._cursor += 1
        content = json.dumps(expected_step.response)
        return CompletionResponse(
            content=content,
            model=self.default_model,
            finish_reason="stop",
        )

    def stream(
        self,
        messages: list[ProviderMessage],
        model: str | None = None,
        options: CompletionOptions | None = None,
    ) -> AsyncIterator[StreamChunk]:
        _ = messages, model, options

        async def _raise() -> AsyncIterator[StreamChunk]:
            raise ReplayDivergenceError("ReplayProvider does not support streaming.")
            yield  # pragma: no cover

        return _raise()

    def supports_streaming(self) -> bool:
        return False

    def supports_tools(self) -> bool:
        return False

    def supports_json_mode(self) -> bool:
        return True

    def get_context_limit(self, model: str | None = None) -> int:
        _ = model
        return 0
