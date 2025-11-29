from __future__ import annotations

import asyncio
import re
import shlex
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Generic, Mapping, Protocol, Sequence, TypeVar

from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field

from jpscripts.core.command_validation import CommandVerdict, validate_command
from jpscripts.core.console import get_logger
from jpscripts.core.mcp_registry import get_tool_registry
from jpscripts.core.config import AppConfig
from jpscripts.core.system import CommandResult, get_sandbox
from jpscripts.core.result import Err
from jpscripts.core.governance import check_compliance, format_violations_for_agent, has_fatal_violations, Violation
from jpscripts.core.result import ToolExecutionError

logger = get_logger(__name__)

ResponseT = TypeVar("ResponseT", bound=BaseModel)
AUDIT_PREFIX = "audit.shell"


class MemoryProtocol(Protocol):
    def query(self, text: str, limit: int = 5) -> list[str]:
        ...

    def save(self, content: str, tags: list[str] | None = None) -> None:
        ...


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
    arguments: dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool")


class AgentResponse(BaseModel):
    """Structured response contract for agent outputs."""

    thought_process: str = Field(..., description="Deep analysis of the problem")
    criticism: str | None = Field(..., description="Self-critique of previous failures")
    tool_call: ToolCall | None = Field(None, description="Tool invocation request")
    file_patch: str | None = Field(None, description="Unified diff to apply (optional)")
    final_message: str | None = Field(None, description="Response to user if no action needed")


class AgentTraceStep(BaseModel):
    timestamp: str
    agent_persona: str
    input_history: list[dict[str, str]]
    response: dict[str, Any]
    tool_output: str | None = None


class TraceRecorder:
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
                logger.warning("TraceRecorder falling back to %s due to permission issues", fallback)
                return fallback
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug("Failed to create fallback trace dir %s: %s", fallback, exc)
                raise

    async def append(self, step: AgentTraceStep) -> None:
        payload = step.model_dump_json()
        await asyncio.to_thread(self._write_line, payload)

    def _write_line(self, line: str) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _clean_json_payload(text: str) -> str:
    """Extract JSON content from raw agent output, tolerating code fences and stray prose."""
    stripped = text.strip()
    if not stripped:
        return stripped

    fence = re.search(r"```json\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip()
        if candidate:
            return candidate

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = stripped[start : end + 1].strip()
        if candidate:
            return candidate

    return stripped


def parse_agent_response(payload: str) -> AgentResponse:
    """Parse and validate a JSON agent response."""
    cleaned = _clean_json_payload(payload)
    return AgentResponse.model_validate_json(cleaned)


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
        self._tools: Mapping[str, Callable[..., Awaitable[str]]] = tools if tools is not None else get_tool_registry()
        self._memory = memory
        self._template_root = template_root
        self._trace_recorder = TraceRecorder(trace_dir or Path.home() / ".jpscripts" / "traces")
        self._workspace_root = workspace_root
        self._governance_enabled = governance_enabled

    async def _render_prompt(self, history: Sequence[Message]) -> PreparedPrompt:
        return await self._prompt_builder(history)

    async def step(self, history: list[Message]) -> ResponseT:
        prepared = await self._render_prompt(history)
        raw = await self._fetch_response(prepared)
        response = self._parser(raw)

        # Apply governance check if enabled and workspace_root is set
        if self._governance_enabled and self._workspace_root is not None:
            response = await self._enforce_governance(response, history)

        await self._record_trace(history, response)
        return response

    async def _enforce_governance(
        self, response: ResponseT, history: list[Message]
    ) -> ResponseT:
        """Check response for constitutional violations and request corrections.

        Implements hard-gating strategy:
        - Fatal violations (SHELL_TRUE, OS_SYSTEM, BARE_EXCEPT) DROP the patch
        - Non-fatal violations trigger retry with agent feedback
        - Maximum 3 retry attempts before raising ToolExecutionError

        Args:
            response: The parsed agent response
            history: Conversation history for context

        Returns:
            Original response if compliant, or corrected response after retry

        Raises:
            ToolExecutionError: If fatal violations persist after max retries
        """
        max_retries = 3
        current_response = response
        current_history = list(history)

        for attempt in range(max_retries):
            # Only check responses with file patches
            if not hasattr(current_response, "file_patch"):
                return current_response

            file_patch = getattr(current_response, "file_patch", None)
            if not file_patch or not self._workspace_root:
                return current_response

            # Check for violations
            violations = check_compliance(str(file_patch), self._workspace_root)
            if not violations:
                return current_response

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
                        for v in violations if v.fatal
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
                current_history = current_history + [governance_message]

                # Re-prompt for correction
                try:
                    prepared = await self._render_prompt(current_history)
                    raw = await self._fetch_response(prepared)
                    current_response = self._parser(raw)
                    continue  # Check the new response
                except Exception as exc:
                    logger.warning("Governance correction failed: %s", exc)
                    raise ToolExecutionError(
                        f"Governance correction failed after fatal violation: {exc}"
                    ) from exc

            # Non-fatal violations: warn and return (allow patch with warnings)
            logger.warning(
                "Non-fatal governance violations detected, proceeding with warnings"
            )
            return current_response

        # Should not reach here, but fail safely
        raise ToolExecutionError(
            f"Governance enforcement exceeded {max_retries} attempts"
        )

    async def _record_trace(self, history: Sequence[Message], response: BaseModel, tool_output: str | None = None) -> None:
        try:
            step = AgentTraceStep(
                timestamp=datetime.now(timezone.utc).isoformat(),
                agent_persona=self.persona,
                input_history=[{"role": msg.role, "content": msg.content} for msg in history],
                response=response.model_dump(),
                tool_output=tool_output,
            )
            await self._trace_recorder.append(step)
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("Failed to record trace: %s", exc)

    async def execute_tool(self, call: ToolCall) -> str:
        """Execute a tool from the unified registry.

        Tools are discovered from jpscripts.mcp.tools and called with
        arguments unpacked as keyword arguments.
        """
        normalized = call.tool.strip().lower()
        if normalized not in self._tools:
            return f"Unknown tool: {call.tool}"

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


async def run_safe_shell(command: str, root: Path, audit_prefix: str, config: AppConfig | None = None) -> str:
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
