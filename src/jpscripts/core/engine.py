from __future__ import annotations

import asyncio
import gzip
import re
import shlex
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar, cast

import tiktoken
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field

from jpscripts.core import runtime, security
from jpscripts.core.command_validation import CommandVerdict, validate_command
from jpscripts.core.config import AppConfig
from jpscripts.core.console import get_logger
from jpscripts.core.cost_tracker import TokenUsage
from jpscripts.core.governance import (
    check_compliance,
    format_violations_for_agent,
    has_fatal_violations,
)
from jpscripts.core.mcp_registry import get_tool_registry
from jpscripts.core.result import Err, ToolExecutionError
from jpscripts.core.runtime import CircuitBreaker
from jpscripts.core.system import CommandResult, get_sandbox


class SpanProtocol(Protocol):
    def set_attribute(self, key: str, value: object) -> None: ...

    def add_event(self, name: str, attributes: Mapping[str, object] | None = None) -> None: ...


class TracerProtocol(Protocol):
    def start_as_current_span(self, name: str) -> AbstractContextManager[SpanProtocol]: ...


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


_otel_trace_module: TraceModuleProtocol | None = None
_otel_resource_cls: type[ResourceProtocol] | None = None
_otel_tracer_provider_cls: type[TracerProviderProtocol] | None = None
_otel_span_processor_cls: type[BatchSpanProcessorProtocol] | None = None
_otel_exporter_cls: type[OTLPSpanExporterProtocol] | None = None
_otel_tracer: TracerProtocol | None = None  # Optional tracer when opentelemetry is installed
_otel_provider_configured = False

logger = get_logger(__name__)

ResponseT = TypeVar("ResponseT", bound=BaseModel)
AUDIT_PREFIX = "audit.shell"
THINKING_PATTERN = re.compile(r"<thinking>(.*?)</thinking>", flags=re.IGNORECASE | re.DOTALL)

# Lazy-loaded tiktoken encoder for accurate token counting
_TOKENIZER: tiktoken.Encoding | None = None


def _get_tokenizer() -> tiktoken.Encoding:
    """Get or initialize the tiktoken encoder (cl100k_base for GPT-4/Claude)."""
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = tiktoken.get_encoding("cl100k_base")
    return _TOKENIZER


class MemoryProtocol(Protocol):
    def query(self, text: str, limit: int = 5) -> list[str]: ...

    def save(self, content: str, tags: list[str] | None = None) -> None: ...


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


class TraceRecorder:
    MAX_TRACE_SIZE = 10 * 1024 * 1024  # 10MB
    ARCHIVE_MAX_AGE_DAYS = 30

    def __init__(self, trace_dir: Path, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or uuid.uuid4().hex
        self.trace_dir = self._ensure_trace_dir(trace_dir)
        self._path = self.trace_dir / f"{self.trace_id}.jsonl"

    @property
    def path(self) -> Path:
        return self._path

    def _ensure_trace_dir(self, trace_dir: Path) -> Path:
        primary = trace_dir.expanduser()
        try:
            primary.mkdir(parents=True, exist_ok=True)
            return primary
        except PermissionError:
            fallback = (Path.cwd() / ".jpscripts" / "traces").resolve()
            try:
                fallback.mkdir(parents=True, exist_ok=True)
                logger.warning(
                    "TraceRecorder falling back to %s due to permission issues", fallback
                )
                return fallback
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug("Failed to create fallback trace dir %s: %s", fallback, exc)
                raise

    async def append(self, step: AgentTraceStep) -> None:
        payload = step.model_dump_json()
        await asyncio.to_thread(self._write_line, payload)

    def _write_line(self, line: str) -> None:
        self._rotate_if_needed()
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def _rotate_if_needed(self) -> None:
        """Compress trace file if it exceeds size limit."""
        if not self._path.exists():
            return
        try:
            if self._path.stat().st_size < self.MAX_TRACE_SIZE:
                return
        except OSError:
            return

        # Compress to .jsonl.gz with timestamp
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        archive_path = self.trace_dir / f"{self.trace_id}.{timestamp}.jsonl.gz"

        try:
            with self._path.open("rb") as f_in, gzip.open(archive_path, "wb") as f_out:
                f_out.writelines(f_in)

            # Truncate original file
            self._path.write_text("")

            # Clean up old archives
            self._cleanup_old_archives()
        except OSError as exc:
            logger.debug("Failed to rotate trace file: %s", exc)

    def _cleanup_old_archives(self) -> None:
        """Delete .jsonl.gz archives older than 30 days."""
        cutoff = datetime.now(UTC) - timedelta(days=self.ARCHIVE_MAX_AGE_DAYS)
        for archive in self.trace_dir.glob("*.jsonl.gz"):
            try:
                mtime = datetime.fromtimestamp(archive.stat().st_mtime, tz=UTC)
                if mtime < cutoff:
                    archive.unlink()
                    logger.debug("Deleted old trace archive: %s", archive)
            except OSError:
                pass  # Ignore cleanup failures


def _extract_balanced_json(text: str) -> str:
    """Extract first complete JSON object using balanced brace matching.

    Properly handles:
    - Nested braces in string values
    - Escape sequences
    - Unmatched braces (falls back to first { to last })
    """
    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    # Fallback: unbalanced braces, return from start to last }
    end = text.rfind("}")
    if end > start:
        return text[start : end + 1]
    return text


def _clean_json_payload(text: str) -> str:
    """Extract JSON content from raw agent output, tolerating code fences and stray prose."""
    stripped = text.strip()
    if not stripped:
        return stripped

    # Try markdown fence first
    fence = re.search(r"```json\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip()
        if candidate:
            return candidate

    # Use balanced brace extraction for proper handling
    return _extract_balanced_json(stripped)


def _split_thought_and_json(payload: str) -> tuple[str, str]:
    """Separate thinking content from JSON payload for strict validation."""
    stripped = payload.strip()
    if not stripped:
        return "", ""

    thinking_match = THINKING_PATTERN.search(stripped)
    if thinking_match:
        preamble = stripped[: thinking_match.start()].strip()
        thinking = thinking_match.group(1).strip()
        thought_parts = [part for part in (preamble, thinking) if part]
        remaining = stripped[thinking_match.end() :].strip()
        json_candidate = _clean_json_payload(remaining or stripped)
        return "\n\n".join(thought_parts), json_candidate

    json_content = _clean_json_payload(stripped)
    if not json_content:
        return stripped, ""

    json_start = stripped.find(json_content)
    preamble = stripped[:json_start].strip() if json_start != -1 else ""
    return preamble, json_content


def parse_agent_response(payload: str) -> AgentResponse:
    """Parse and validate a JSON agent response."""
    thought_content, json_content = _split_thought_and_json(payload)
    response = AgentResponse.model_validate_json(json_content)
    if thought_content:
        response.thought_process = thought_content
    return response


def _approximate_tokens(content: str) -> int:
    """Count tokens using tiktoken for accuracy (with fallback)."""
    if not content:
        return 0
    try:
        return len(_get_tokenizer().encode(content, disallowed_special=()))
    except Exception:
        # Fallback to char/4 estimate if tiktoken fails
        return max(1, len(content) // 4)


def _estimate_token_usage(prompt_text: str, completion_text: str) -> TokenUsage:
    """Token estimate using tiktoken for circuit breaker budget tracking."""
    return TokenUsage(
        prompt_tokens=_approximate_tokens(prompt_text),  # safety: checked
        completion_tokens=_approximate_tokens(completion_text),  # safety: checked
    )


def _extract_patch_paths(file_patch: str, workspace_root: Path) -> list[Path]:
    """Derive touched files from a unified diff."""
    if not file_patch.strip():
        return []

    candidates: set[Path] = set()
    for raw_line in file_patch.splitlines():
        if not raw_line.startswith(("+++ ", "--- ")):
            continue
        try:
            _, path_str = raw_line.split(" ", 1)
        except ValueError:
            continue
        normalized_line = path_str.strip()
        if normalized_line in {"/dev/null", "dev/null", "a/dev/null", "b/dev/null"}:
            continue
        if normalized_line.startswith(("a/", "b/")):
            normalized_line = normalized_line[2:]
        try:
            normalized_path = security.validate_path(
                workspace_root / normalized_line,
                workspace_root,
            )
        except Exception as exc:
            logger.debug("Skipping patch path %s: %s", normalized_line, exc)
            continue
        candidates.add(normalized_path)
    return sorted(candidates)


def _build_black_box_report(
    breaker: CircuitBreaker,
    *,
    usage: TokenUsage,
    files_touched: list[Path],
    persona: str,
    context: str,
) -> str:
    file_lines = "\n".join(f"- {path}" for path in files_touched) if files_touched else "- (none)"
    reason = breaker.last_failure_reason or "Unknown"
    return (
        "=== Black Box Crash Report ===\n"
        f"Persona: {persona}\n"
        f"Context: {context}\n"
        f"Reason: {reason}\n"
        f"Prompt tokens: {usage.prompt_tokens}\n"
        f"Completion tokens: {usage.completion_tokens}\n"
        f"Cost estimate (USD): {breaker.last_cost_estimate}\n"
        f"Cost velocity (USD/min): {breaker.last_cost_velocity}\n"
        f"Max velocity allowed (USD/min): {breaker.max_cost_velocity}\n"
        f"File churn: {breaker.last_file_churn}\n"
        f"Max file churn allowed: {breaker.max_file_churn}\n"
        "Files observed:\n"
        f"{file_lines}"
    )


def _load_otel_deps() -> tuple[
    TraceModuleProtocol | None,
    type[ResourceProtocol] | None,
    type[TracerProviderProtocol] | None,
    type[BatchSpanProcessorProtocol] | None,
    type[OTLPSpanExporterProtocol] | None,
]:
    """Dynamically import opentelemetry components if available."""
    try:
        import importlib

        trace_mod = importlib.import_module("opentelemetry.trace")
        resources_mod = importlib.import_module("opentelemetry.sdk.resources")
        trace_sdk_mod = importlib.import_module("opentelemetry.sdk.trace")
        trace_export_mod = importlib.import_module("opentelemetry.sdk.trace.export")

        exporter_cls = None
        try:
            exporter_mod = importlib.import_module(
                "opentelemetry.exporter.otlp.proto.http.trace_exporter"
            )
            exporter_cls = getattr(exporter_mod, "OTLPSpanExporter", None)
        except ImportError:
            exporter_cls = None

        return (
            cast(TraceModuleProtocol, trace_mod),
            cast(type[ResourceProtocol], getattr(resources_mod, "Resource", None)),
            cast(type[TracerProviderProtocol], getattr(trace_sdk_mod, "TracerProvider", None)),
            cast(
                type[BatchSpanProcessorProtocol],
                getattr(trace_export_mod, "BatchSpanProcessor", None),
            ),
            cast(type[OTLPSpanExporterProtocol], exporter_cls) if exporter_cls else None,
        )
    except ImportError:
        return None, None, None, None, None
    except Exception:
        return None, None, None, None, None


def _get_tracer() -> TracerProtocol | None:
    """Lazily configure and return an OTLP-capable tracer."""
    global _otel_tracer, _otel_provider_configured
    global _otel_trace_module, _otel_resource_cls, _otel_tracer_provider_cls
    global _otel_span_processor_cls, _otel_exporter_cls

    if _otel_trace_module is None or _otel_tracer_provider_cls is None:
        (
            _otel_trace_module,
            _otel_resource_cls,
            _otel_tracer_provider_cls,
            _otel_span_processor_cls,
            _otel_exporter_cls,
        ) = _load_otel_deps()

    if _otel_trace_module is None or _otel_tracer_provider_cls is None:
        return None

    try:
        runtime_ctx = runtime.get_runtime()
    except Exception:
        return None

    config = runtime_ctx.config
    if not getattr(config, "otel_export_enabled", False):
        return None

    if _otel_tracer is not None:
        return _otel_tracer

    try:
        resource: ResourceProtocol | None = (
            _otel_resource_cls.create({"service.name": config.otel_service_name})
            if _otel_resource_cls
            else None
        )
        provider: TracerProviderProtocol = (
            _otel_tracer_provider_cls(resource=resource)
            if resource is not None
            else _otel_tracer_provider_cls()
        )
        if _otel_span_processor_cls is not None and _otel_exporter_cls is not None:
            try:
                exporter: OTLPSpanExporterProtocol = (
                    _otel_exporter_cls(endpoint=config.otel_endpoint)
                    if config.otel_endpoint
                    else _otel_exporter_cls()
                )
                provider.add_span_processor(_otel_span_processor_cls(exporter))
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug("Failed to configure OTLP exporter: %s", exc)
        if not _otel_provider_configured:
            _otel_trace_module.set_tracer_provider(provider)
            _otel_provider_configured = True
        _otel_tracer = _otel_trace_module.get_tracer(config.otel_service_name or "jpscripts")
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug("Failed to initialize tracer: %s", exc)
        return None

    return _otel_tracer


class AgentEngine(Generic[ResponseT]):
    def __init__(
        self,
        *,
        persona: str,
        model: str,
        prompt_builder: Callable[[Sequence[Message]], Awaitable[PreparedPrompt]],
        fetch_response: Callable[[PreparedPrompt], Awaitable[str]],
        parser: Callable[[str], ResponseT],
        tools: Mapping[str, Callable[..., Awaitable[str]]] | None = None,
        memory: MemoryProtocol | None = None,
        template_root: Path | None = None,
        trace_dir: Path | None = None,
        workspace_root: Path | None = None,
        governance_enabled: bool = True,
    ) -> None:
        self.persona = persona
        self.model = model
        self._prompt_builder = prompt_builder
        self._fetch_response = fetch_response
        self._parser = parser
        # Use unified tool registry if no tools provided
        self._tools: Mapping[str, Callable[..., Awaitable[str]]] = (
            tools if tools is not None else get_tool_registry()
        )
        self._memory = memory
        self._template_root = template_root
        self._trace_recorder = TraceRecorder(trace_dir or Path.home() / ".jpscripts" / "traces")
        self._workspace_root = workspace_root
        self._governance_enabled = governance_enabled
        self._last_usage_snapshot: TokenUsage | None = None
        self._last_files_touched: list[Path] = []

    async def _render_prompt(self, history: Sequence[Message]) -> PreparedPrompt:
        return await self._prompt_builder(history)

    async def step(self, history: list[Message]) -> ResponseT:
        prepared = await self._render_prompt(history)
        raw = await self._fetch_response(prepared)
        response = self._parser(raw)

        # Apply governance check if enabled and workspace_root is set
        if self._governance_enabled and self._workspace_root is not None:
            response, prepared, raw = await self._enforce_governance(
                response,
                history,
                prepared,
                raw,
            )

        usage_snapshot = _estimate_token_usage(prepared.prompt, raw)
        files_touched = self._infer_files_touched(response)
        self._last_usage_snapshot = usage_snapshot
        self._last_files_touched = files_touched

        self._enforce_circuit_breaker(
            usage=usage_snapshot,
            files_touched=files_touched,
            context="agent_response",
        )

        await self._record_trace(history, response)
        return response

    async def _enforce_governance(
        self,
        response: ResponseT,
        history: list[Message],
        prepared: PreparedPrompt,
        raw_response: str,
    ) -> tuple[ResponseT, PreparedPrompt, str]:
        """Check response for constitutional violations and request corrections.

        Implements hard-gating strategy:
        - Fatal violations (SHELL_TRUE, OS_SYSTEM, BARE_EXCEPT) DROP the patch
        - Non-fatal violations trigger retry with agent feedback
        - Maximum 3 retry attempts before raising ToolExecutionError

        Args:
            response: The parsed agent response
            history: Conversation history for context
            prepared: Prepared prompt issued for this attempt
            raw_response: Raw model output tied to the parsed response

        Returns:
            Tuple of (response, prepared_prompt, raw_response) representing the final compliant attempt

        Raises:
            ToolExecutionError: If fatal violations persist after max retries
        """
        max_retries = 3
        current_response = response
        current_history = list(history)
        current_prepared = prepared
        current_raw = raw_response

        for attempt in range(max_retries):
            # Only check responses with file patches
            if not hasattr(current_response, "file_patch"):
                return current_response, current_prepared, current_raw

            file_patch = getattr(current_response, "file_patch", None)
            if not file_patch or not self._workspace_root:
                return current_response, current_prepared, current_raw

            # Check for violations
            violations = check_compliance(str(file_patch), self._workspace_root)
            if not violations:
                return current_response, current_prepared, current_raw

            # Log violations
            error_count = sum(1 for v in violations if v.severity == "error")
            warning_count = len(violations) - error_count
            fatal_count = sum(1 for v in violations if v.fatal)
            logger.warning(
                "Governance violations detected (attempt %d/%d): %d errors (%d fatal), %d warnings",
                attempt + 1,
                max_retries,
                error_count,
                fatal_count,
                warning_count,
            )

            # Check for fatal violations - DROP the patch
            if has_fatal_violations(violations):
                logger.error(
                    "Fatal governance violation detected - patch DROPPED (attempt %d/%d)",
                    attempt + 1,
                    max_retries,
                )

                # Last attempt - raise error
                if attempt >= max_retries - 1:
                    fatal_msgs = [
                        f"{v.type.name} at {v.file.name}:{v.line}: {v.message}"
                        for v in violations
                        if v.fatal
                    ]
                    raise ToolExecutionError(
                        f"Fatal governance violations after {max_retries} attempts:\n"
                        + "\n".join(fatal_msgs)
                    )

                # Format feedback and inject into history for retry
                feedback = format_violations_for_agent(violations)
                governance_message = Message(
                    role="system",
                    content=(
                        f"<GovernanceViolation severity='FATAL'>\n"
                        f"Your patch was REJECTED and NOT APPLIED due to fatal violations.\n"
                        f"{feedback}\n"
                        f"</GovernanceViolation>"
                    ),
                )
                current_history = [*current_history, governance_message]

                # Re-prompt for correction
                try:
                    current_prepared = await self._render_prompt(current_history)
                    current_raw = await self._fetch_response(current_prepared)
                    current_response = self._parser(current_raw)
                    continue  # Check the new response
                except Exception as exc:
                    logger.warning("Governance correction failed: %s", exc)
                    raise ToolExecutionError(
                        f"Governance correction failed after fatal violation: {exc}"
                    ) from exc

            # Non-fatal violations: warn and return (allow patch with warnings)
            logger.warning("Non-fatal governance violations detected, proceeding with warnings")
            return current_response, current_prepared, current_raw

        # Should not reach here, but fail safely
        raise ToolExecutionError(f"Governance enforcement exceeded {max_retries} attempts")

    async def _record_trace(
        self, history: Sequence[Message], response: BaseModel, tool_output: str | None = None
    ) -> None:
        try:
            step = AgentTraceStep(
                timestamp=datetime.now(UTC).isoformat(),
                agent_persona=self.persona,
                input_history=[{"role": msg.role, "content": msg.content} for msg in history],
                response=response.model_dump(),
                tool_output=tool_output,
            )
            await self._trace_recorder.append(step)
            tracer = _get_tracer()
            if tracer is not None:
                files_touched = [str(path) for path in self._last_files_touched]
                usage_snapshot = self._last_usage_snapshot
                with tracer.start_as_current_span("agent.turn") as span:
                    span.set_attribute("agent.persona", self.persona)
                    if files_touched:
                        span.set_attribute("code.files_touched", files_touched)
                    if usage_snapshot is not None:
                        span.set_attribute("usage.prompt_tokens", usage_snapshot.prompt_tokens)
                        span.set_attribute(
                            "usage.completion_tokens", usage_snapshot.completion_tokens
                        )
                        span.set_attribute("usage.total_tokens", usage_snapshot.total_tokens)
                    tool_call = getattr(response, "tool_call", None)
                    if tool_call is not None:
                        span.add_event(
                            "tool_call",
                            {
                                "tool_call": tool_call.model_dump()
                                if hasattr(tool_call, "model_dump")
                                else str(tool_call)
                            },
                        )
                    if tool_output:
                        span.add_event("tool_output", {"output": tool_output})
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("Failed to record trace: %s", exc)

    def _infer_files_touched(self, response: BaseModel) -> list[Path]:
        if not hasattr(response, "file_patch"):
            return []

        file_patch = getattr(response, "file_patch", None)
        if not file_patch or self._workspace_root is None:
            return []

        return _extract_patch_paths(str(file_patch), self._workspace_root)

    def _enforce_circuit_breaker(
        self,
        *,
        usage: TokenUsage,
        files_touched: list[Path],
        context: str,
    ) -> None:
        breaker = runtime.get_circuit_breaker()
        if breaker.check_health(usage, files_touched):
            return

        report = _build_black_box_report(
            breaker,
            usage=usage,
            files_touched=files_touched,
            persona=self.persona,
            context=context,
        )
        logger.error("Circuit breaker triggered: %s", breaker.last_failure_reason)
        raise SafetyLockdownError(report)

    async def execute_tool(self, call: ToolCall) -> str:
        """Execute a tool from the unified registry.

        Tools are discovered from jpscripts.mcp.tools and called with
        arguments unpacked as keyword arguments.
        """
        normalized = call.tool.strip().lower()
        if normalized not in self._tools:
            return f"Unknown tool: {call.tool}"

        usage = self._last_usage_snapshot or _estimate_token_usage("", "")
        files_touched = list(self._last_files_touched)
        self._enforce_circuit_breaker(
            usage=usage,
            files_touched=files_touched,
            context=f"tool:{normalized}",
        )

        try:
            # Call tool with unpacked kwargs (tools have proper signatures)
            return await self._tools[normalized](**call.arguments)
        except TypeError as exc:
            # Handle argument mismatch errors gracefully
            return f"Tool '{call.tool}' argument error: {exc}"
        except Exception as exc:
            return f"Tool '{call.tool}' failed: {exc}"


def load_template_environment(template_root: Path) -> Environment:
    return Environment(loader=FileSystemLoader(str(template_root)), autoescape=False)


async def run_safe_shell(
    command: str, root: Path, audit_prefix: str, config: AppConfig | None = None
) -> str:
    """
    Shared safe shell runner for AgentEngine and MCP.

    Uses tokenized command validation to enforce:
    - Allowlisted binaries only (read-only operations)
    - No shell metacharacters (pipes, redirects, etc.)
    - Path validation (no workspace escape)
    - Forbidden flag detection

    Args:
        command: The shell command to execute
        root: Workspace root directory (commands must stay within)
        audit_prefix: Prefix for audit log entries

    Returns:
        Command output on success, error message on failure
    """
    # Use tokenized validation instead of regex
    verdict, reason = validate_command(command, root)

    if verdict != CommandVerdict.ALLOWED:
        logger.warning(
            "%s.reject verdict=%s reason=%r command=%r",
            audit_prefix,
            verdict.name,
            reason,
            command,
        )
        # Map verdict to user-friendly error message
        if verdict == CommandVerdict.BLOCKED_FORBIDDEN:
            return f"SecurityError: {reason}"
        if verdict == CommandVerdict.BLOCKED_NOT_ALLOWLISTED:
            return f"SecurityError: Command not permitted by policy. {reason}"
        if verdict == CommandVerdict.BLOCKED_PATH_ESCAPE:
            return f"SecurityError: {reason}"
        if verdict == CommandVerdict.BLOCKED_DANGEROUS_FLAG:
            return f"SecurityError: {reason}"
        if verdict == CommandVerdict.BLOCKED_METACHAR:
            return f"SecurityError: {reason}"
        if verdict == CommandVerdict.BLOCKED_UNPARSEABLE:
            return f"Unable to parse command; simplify quoting. ({reason})"
        return f"SecurityError: {reason}"

    # Parse command for execution
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        logger.warning("%s.reject parse_error=%s", audit_prefix, exc)
        return f"Unable to parse command; simplify quoting. ({exc})"

    if not tokens:
        return "Invalid command argument."

    runner = get_sandbox(config)
    run_result = await runner.run_command(tokens, root, env=None)
    if isinstance(run_result, Err):
        logger.warning("%s.reject runner_error=%s", audit_prefix, run_result.error)
        return f"Failed to run command: {run_result.error}"

    result: CommandResult = run_result.value
    if result.returncode != 0:
        logger.warning("%s.fail code=%s cmd=%r", audit_prefix, result.returncode, command)
        return f"Command failed with exit code {result.returncode}"

    combined = (result.stdout + result.stderr).strip()
    return combined or "Command produced no output."
