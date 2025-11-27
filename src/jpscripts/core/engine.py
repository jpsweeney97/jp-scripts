from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Generic, Mapping, Protocol, Sequence, TypeVar

from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from pydantic import BaseModel, Field, ValidationError

from jpscripts.core import security
from jpscripts.core.console import get_logger
from jpscripts.core.context import smart_read_context

logger = get_logger(__name__)

ResponseT = TypeVar("ResponseT", bound=BaseModel)


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
    tool_call: ToolCall | None = Field(None, description="Tool invocation request")
    file_patch: str | None = Field(None, description="Unified diff to apply (optional)")
    final_message: str | None = Field(None, description="Response to user if no action needed")


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
        tools: Mapping[str, Callable[[dict[str, Any]], Awaitable[str]]] | None = None,
        memory: MemoryProtocol | None = None,
        template_root: Path | None = None,
    ) -> None:
        self.persona = persona
        self.model = model
        self._prompt_builder = prompt_builder
        self._fetch_response = fetch_response
        self._parser = parser
        self._tools = tools or {}
        self._memory = memory
        self._template_root = template_root

    async def _render_prompt(self, history: Sequence[Message]) -> PreparedPrompt:
        return await self._prompt_builder(history)

    async def step(self, history: list[Message]) -> ResponseT:
        prepared = await self._render_prompt(history)
        raw = await self._fetch_response(prepared)
        return self._parser(raw)

    async def execute_tool(self, call: ToolCall) -> str:
        """
        Execute a tool with security constraints. Defaults include read_file and run_shell.
        """
        normalized = call.tool.strip().lower()
        if normalized in self._tools:
            try:
                return await self._tools[normalized](call.arguments)
            except Exception as exc:
                return f"Tool '{call.tool}' failed: {exc}"

        if normalized == "read_file":
            return await self._read_file_tool(call.arguments)
        if normalized == "run_shell":
            return await self._run_shell_tool(call.arguments)
        return f"Unknown tool: {call.tool}"

    async def _read_file_tool(self, args: dict[str, Any]) -> str:
        path_arg = args.get("path")
        if not isinstance(path_arg, str):
            return "Invalid path argument."
        target = Path(path_arg)
        root = self._template_root or Path.cwd()
        candidate = target if target.is_absolute() else root / target
        try:
            safe_target = security.validate_path(candidate, root)
        except Exception as exc:
            return f"Security error validating path: {exc}"

        try:
            return await asyncio.to_thread(smart_read_context, safe_target)
        except FileNotFoundError:
            return f"File not found: {safe_target}"
        except Exception as exc:  # pragma: no cover - defensive
            return f"Failed to read file {safe_target}: {exc}"

    async def _run_shell_tool(self, args: dict[str, Any]) -> str:
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            return "Invalid command argument."
        forbidden = re.compile(r"(rm|mv|cp|>|\\||sudo|chmod)", flags=re.IGNORECASE)
        if forbidden.search(command):
            return "SecurityError: Command contains forbidden operations."
        allowed = re.compile(r"^(ls|grep|find|cat|git status|git diff|git log)")
        if not allowed.match(command.strip()):
            return "SecurityError: Command not permitted by policy."

        root = self._template_root or Path.cwd()
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as exc:  # pragma: no cover - defensive
            return f"Failed to run command: {exc}"

        stdout_bytes, stderr_bytes = await proc.communicate()
        stdout = stdout_bytes.decode(errors="replace")
        stderr = stderr_bytes.decode(errors="replace")
        if proc.returncode != 0:
            return f"Command failed with exit code {proc.returncode}"
        return (stdout + stderr).strip() or "Command produced no output."


def load_template_environment(template_root: Path) -> Environment:
    return Environment(loader=FileSystemLoader(str(template_root)), autoescape=False)
