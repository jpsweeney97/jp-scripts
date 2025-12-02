"""Agent data models and protocol definitions.

This module contains the core dataclasses, protocols, and type definitions
used throughout the agent subsystem, including:
- Response and message types
- OpenTelemetry protocol types
- Event types and configuration
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

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
# Core Data Models (from engine/models.py)
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
# Event Types and Configuration (from agent/types.py)
# -----------------------------------------------------------------------------


# Re-export from core for backward compatibility
from jpscripts.core.errors import SecurityError

# Type aliases for async fetchers
PatchFetcher = Callable[["PreparedPrompt"], Awaitable[str]]
ResponseFetcher = Callable[["PreparedPrompt"], Awaitable[str]]


class EventKind(Enum):
    """Types of events from the repair loop."""

    ATTEMPT_START = "attempt_start"
    COMMAND_SUCCESS = "command_success"
    COMMAND_FAILED = "command_failed"
    TOOL_CALL = "tool_call"
    TOOL_OUTPUT = "tool_output"
    PATCH_PROPOSED = "patch_proposed"
    PATCH_APPLIED = "patch_applied"
    SYNTAX_ERROR = "syntax_error"
    DUPLICATE_PATCH = "duplicate_patch"
    LOOP_DETECTED = "loop_detected"
    VALIDATION_ERROR = "validation_error"
    NO_PATCH = "no_patch"
    REVERTING = "reverting"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """Structured event from repair loop operations."""

    kind: EventKind
    message: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class RepairLoopConfig:
    """Configuration for the autonomous repair loop.

    Groups optional parameters for run_repair_loop to reduce
    function signature complexity.
    """

    attach_recent: bool = False
    """Attach recently modified files to context."""

    include_diff: bool = True
    """Include git diff in context."""

    auto_archive: bool = True
    """Archive successful fixes to memory."""

    max_retries: int = 3
    """Maximum repair attempts before giving up."""

    keep_failed: bool = False
    """Keep changes even if repair loop fails."""

    web_access: bool = False
    """Enable web search for context."""


# -----------------------------------------------------------------------------
# Exports
# -----------------------------------------------------------------------------

__all__ = [
    # Event types (from agent/types.py)
    "AgentEvent",
    # Core models
    "AgentResponse",
    "AgentTraceStep",
    # OpenTelemetry protocols
    "BatchSpanProcessorProtocol",
    "EventKind",
    "MemoryProtocol",
    "Message",
    "OTLPSpanExporterProtocol",
    "PatchFetcher",
    "PreparedPrompt",
    "RepairLoopConfig",
    "ResourceProtocol",
    "ResponseFetcher",
    "ResponseT",
    "SafetyLockdownError",
    "SecurityError",
    "SpanProcessorProtocol",
    "SpanProtocol",
    "ToolCall",
    "TraceModuleProtocol",
    "TracerProtocol",
    "TracerProviderProtocol",
]
