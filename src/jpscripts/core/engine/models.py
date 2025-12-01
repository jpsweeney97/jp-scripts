"""Engine data models and protocol definitions.

This module contains the core dataclasses, protocols, and type definitions
used throughout the engine subsystem.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # pyright: ignore[reportMissingImports]
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource  # pyright: ignore[reportMissingImports]
    from opentelemetry.sdk.trace import TracerProvider  # pyright: ignore[reportMissingImports]
    from opentelemetry.sdk.trace.export import (  # pyright: ignore[reportMissingImports]
        BatchSpanProcessor,
    )
else:  # pragma: no cover - optional dependency
    OTLPSpanExporter = None
    Resource = None
    TracerProvider = None
    BatchSpanProcessor = None


# -----------------------------------------------------------------------------
# Response Type Variable
# -----------------------------------------------------------------------------

ResponseT = TypeVar("ResponseT", bound=BaseModel)


# -----------------------------------------------------------------------------
# OpenTelemetry Protocol Types
# -----------------------------------------------------------------------------


class SpanProtocol(Protocol):
    def set_attribute(self, key: str, value: object) -> None: ...

    def add_event(self, name: str, attributes: Mapping[str, object] | None = None) -> None: ...


class TracerProtocol(Protocol):
    def start_as_current_span(self, name: str) -> AbstractContextManager[SpanProtocol]: ...


class ResourceProtocol(Protocol):
    @classmethod
    def create(cls, attributes: Mapping[str, object]) -> ResourceProtocol: ...


class TracerProviderProtocol(Protocol):
    def __init__(self, resource: ResourceProtocol | None = None) -> None: ...

    def add_span_processor(self, processor: SpanProcessorProtocol) -> None: ...


class SpanProcessorProtocol(Protocol): ...


class BatchSpanProcessorProtocol(SpanProcessorProtocol, Protocol):
    def __init__(self, exporter: object) -> None: ...


class OTLPSpanExporterProtocol(Protocol):
    def __init__(self, endpoint: str | None = None) -> None: ...


class TraceModuleProtocol(Protocol):
    def set_tracer_provider(self, provider: TracerProviderProtocol) -> None: ...

    def get_tracer(self, name: str) -> TracerProtocol: ...


# -----------------------------------------------------------------------------
# Memory Protocol
# -----------------------------------------------------------------------------


class MemoryProtocol(Protocol):
    def query(self, text: str, limit: int = 5) -> list[str]: ...

    def save(self, content: str, tags: list[str] | None = None) -> None: ...


# -----------------------------------------------------------------------------
# Core Data Models
# -----------------------------------------------------------------------------


@dataclass
class PreparedPrompt:
    prompt: str
    attached_files: list[Path]
    temperature: float | None = None
    reasoning_effort: str | None = None


@dataclass
class Message:
    role: str
    content: str


class ToolCall(BaseModel):
    tool: str = Field(..., description="Name of the tool to invoke")
    arguments: dict[str, object] = Field(default_factory=dict, description="Arguments for the tool")


class AgentResponse(BaseModel):
    """Structured response contract for agent outputs."""

    thought_process: str = Field(..., description="Deep analysis of the problem")
    criticism: str | None = Field(..., description="Self-critique of previous failures")
    tool_call: ToolCall | None = Field(None, description="Tool invocation request")
    file_patch: str | None = Field(None, description="Unified diff to apply (optional)")
    final_message: str | None = Field(None, description="Response to user if no action needed")


class SafetyLockdownError(RuntimeError):
    """Raised when the circuit breaker halts an agent turn."""

    def __init__(self, report: str) -> None:
        self.report = report
        super().__init__(f"SafetyLockdownError triggered\n{report}")


class AgentTraceStep(BaseModel):
    timestamp: str
    agent_persona: str
    input_history: list[dict[str, str]]
    response: dict[str, object]
    tool_output: str | None = None


# -----------------------------------------------------------------------------
# Exports
# -----------------------------------------------------------------------------

__all__ = [
    "AgentResponse",
    "AgentTraceStep",
    "BatchSpanProcessorProtocol",
    "MemoryProtocol",
    "Message",
    "OTLPSpanExporterProtocol",
    "PreparedPrompt",
    "ResourceProtocol",
    "ResponseT",
    "SafetyLockdownError",
    "SpanProcessorProtocol",
    "SpanProtocol",
    "ToolCall",
    "TraceModuleProtocol",
    "TracerProtocol",
    "TracerProviderProtocol",
]
