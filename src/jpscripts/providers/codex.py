"""
Codex CLI provider adapter.

.. deprecated::
    NOTICE: This provider wraps the external Codex CLI binary. It is a legacy
    adapter and will be replaced by the native Python SDK in a future release.
    New integrations should prefer direct API providers.

This module wraps the Codex CLI binary to provide the LLMProvider interface.
It maintains backward compatibility with the existing jp agent workflow
while enabling gradual migration to direct API providers.

Usage:
    from jpscripts.providers.codex import CodexProvider

    provider = CodexProvider(config)
    response = await provider.complete(
        messages=[Message(role="user", content="Hello")],
        model="o1",
    )

Note:
    This provider requires the Codex CLI to be installed and available in PATH.
    Install via: npm install -g @openai/codex
"""

from __future__ import annotations

import asyncio
import json
import shutil
import warnings
from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any

from jpscripts.core.console import get_logger
from jpscripts.providers import (
    BaseLLMProvider,
    CompletionOptions,
    CompletionResponse,
    Message,
    ProviderError,
    ProviderType,
    StreamChunk,
    ToolCall,
)

if TYPE_CHECKING:
    from jpscripts.core.config import AppConfig

logger = get_logger(__name__)

# Codex supports OpenAI models
CODEX_AVAILABLE_MODELS: tuple[str, ...] = (
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "o1",
    "o1-mini",
    "o3-mini",
)

CODEX_CONTEXT_LIMITS: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "o1": 200_000,
    "o1-mini": 128_000,
    "o3-mini": 200_000,
}


class CodexNotFoundError(ProviderError):
    """Raised when the Codex CLI binary is not found."""

    def __init__(self) -> None:
        super().__init__("Codex CLI not found. Install via: npm install -g @openai/codex")


def _find_codex_binary() -> str | None:
    """Find the Codex CLI binary in PATH."""
    return shutil.which("codex")


def _coerce_tool_args(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract tool arguments from an event payload as a dict."""
    candidate = payload.get("arguments") or payload.get("input")
    if isinstance(candidate, dict):
        return candidate

    if candidate is not None:
        logger.debug("Ignoring non-dict tool arguments: %r", candidate)

    return {}


def _build_codex_command(
    codex_bin: str,
    model: str,
    prompt: str,
    *,
    full_auto: bool = False,
    web: bool = False,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
) -> list[str]:
    """Build the Codex CLI command.

    Args:
        codex_bin: Path to Codex binary
        model: Model ID to use
        prompt: The prompt text
        full_auto: Run without confirmation
        web: Enable web search
        temperature: Temperature setting
        reasoning_effort: Reasoning effort for o1/o3 models

    Returns:
        Command as list of strings
    """
    cmd = [codex_bin, "exec", "--json", "--model", model]

    if reasoning_effort:
        cmd.extend(["-c", f"reasoning.effort={reasoning_effort}"])

    if temperature is not None:
        cmd.extend(["-c", f"temperature={temperature}"])

    if web:
        cmd.append("--search")

    if full_auto:
        cmd.append("--full-auto")

    cmd.append(prompt)
    return cmd


def _format_messages_for_codex(
    messages: list[Message],
    system_prompt: str | None = None,
) -> str:
    """Format messages into a single prompt string for Codex CLI.

    Codex CLI takes a single prompt string, so we need to format
    the conversation history appropriately.
    """
    parts: list[str] = []

    # Add system prompt if provided
    if system_prompt:
        parts.append(f"[System]\n{system_prompt}\n")

    for msg in messages:
        if msg.role == "system":
            parts.append(f"[System]\n{msg.content}\n")
        elif msg.role == "user":
            parts.append(f"[User]\n{msg.content}\n")
        elif msg.role == "assistant":
            parts.append(f"[Assistant]\n{msg.content}\n")

    return "\n".join(parts)


class CodexProvider(BaseLLMProvider):
    """Codex CLI provider adapter.

    This provider wraps the Codex CLI binary, providing the same
    interface as direct API providers while leveraging Codex's
    built-in features like tool execution and web search.

    Attributes:
        full_auto: If True, run Codex in full-auto mode (no confirmations)
        web_enabled: If True, enable web search capability
    """

    def __init__(
        self,
        config: AppConfig,
        *,
        full_auto: bool = False,
        web_enabled: bool = False,
    ) -> None:
        super().__init__(config)
        self._full_auto = full_auto
        self._web_enabled = web_enabled
        self._codex_bin: str | None = None

        # Emit formal deprecation warning
        warnings.warn(
            "CodexProvider is deprecated for jp fix. Use --provider anthropic or "
            "--provider openai for native API access. The Codex CLI wrapper will be "
            "removed in jp-scripts 3.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.warning(
            "DEPRECATION: CodexProvider wraps the external Codex CLI binary. "
            "This adapter is deprecated and will be removed in a future release. "
            "Use native providers (anthropic, openai) instead."
        )

    def _get_codex_binary(self) -> str:
        """Get the Codex binary path, caching the result."""
        if self._codex_bin is None:
            self._codex_bin = _find_codex_binary()
            if self._codex_bin is None:
                raise CodexNotFoundError()
        return self._codex_bin

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.CODEX

    @property
    def default_model(self) -> str:
        return "o1"

    @property
    def available_models(self) -> tuple[str, ...]:
        return CODEX_AVAILABLE_MODELS

    def get_context_limit(self, model: str | None = None) -> int:
        model_id = model or self.default_model
        return CODEX_CONTEXT_LIMITS.get(model_id, 128_000)

    def supports_streaming(self) -> bool:
        """Codex CLI provides streaming via JSON events."""
        return True

    def supports_tools(self) -> bool:
        """Codex has built-in tool support."""
        return True

    def supports_json_mode(self) -> bool:
        """Codex outputs JSON events but doesn't have explicit JSON mode."""
        return False

    async def complete(
        self,
        messages: list[Message],
        model: str | None = None,
        options: CompletionOptions | None = None,
    ) -> CompletionResponse:
        """Send a completion request via Codex CLI."""
        codex_bin = self._get_codex_binary()
        opts = options or CompletionOptions()
        model_id = model or self.default_model

        # Format messages into prompt
        prompt = _format_messages_for_codex(messages, opts.system_prompt)

        # Build command
        cmd = _build_codex_command(
            codex_bin,
            model_id,
            prompt,
            full_auto=self._full_auto,
            web=self._web_enabled,
            temperature=opts.temperature,
            reasoning_effort=opts.reasoning_effort or "high",
        )

        # Execute Codex
        assistant_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        raw_fallback_lines: list[str] = []

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise CodexNotFoundError() from exc
        except Exception as exc:
            raise ProviderError(f"Failed to start Codex: {exc}") from exc

        if proc.stdout is None:
            raise ProviderError("Codex process has no stdout")

        # Process JSON events from stdout
        async for raw_line in proc.stdout:
            line = raw_line.decode(errors="replace").strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                raw_fallback_lines.append(line)
                logger.debug("Non-JSON line from Codex: %s", line[:100])
                continue

            data: dict[str, Any] = event.get("data") or {}
            event_type = event.get("event") or event.get("type")

            # Extract assistant messages
            message = (
                data.get("assistant_message")
                or event.get("assistant_message")
                or data.get("message")
            )
            if isinstance(message, str) and message.strip():
                assistant_parts.append(message.strip())

            # Extract tool calls
            if event_type == "tool.call":
                tool_name = data.get("name") or data.get("tool")
                tool_args = _coerce_tool_args(data)
                if tool_name:
                    tool_calls.append(
                        ToolCall(
                            id=data.get("id", f"call_{len(tool_calls)}"),
                            name=tool_name,
                            arguments=tool_args,
                        )
                    )

        await proc.wait()

        # Check for errors
        if proc.stderr:
            stderr_text = (await proc.stderr.read()).decode(errors="replace").strip()
            if stderr_text and proc.returncode != 0:
                raise ProviderError(f"Codex error: {stderr_text}")

        # Fallback if no structured content was extracted
        if not assistant_parts and raw_fallback_lines:
            assistant_parts = raw_fallback_lines
            logger.warning("Codex output contained no parseable JSON events; using raw output")

        content = "\n\n".join(assistant_parts)

        return CompletionResponse(
            content=content,
            model=model_id,
            finish_reason="stop" if proc.returncode == 0 else "error",
            tool_calls=tool_calls,
            usage=None,  # Codex CLI doesn't report usage
            raw_response=None,
        )

    async def stream(
        self,
        messages: list[Message],
        model: str | None = None,
        options: CompletionOptions | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream completion response from Codex CLI.

        Codex outputs JSON events as lines, which we convert to StreamChunks.
        """
        codex_bin = self._get_codex_binary()
        opts = options or CompletionOptions()
        model_id = model or self.default_model

        prompt = _format_messages_for_codex(messages, opts.system_prompt)

        cmd = _build_codex_command(
            codex_bin,
            model_id,
            prompt,
            full_auto=self._full_auto,
            web=self._web_enabled,
            temperature=opts.temperature,
            reasoning_effort=opts.reasoning_effort or "high",
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise CodexNotFoundError() from exc
        except Exception as exc:
            raise ProviderError(f"Failed to start Codex: {exc}") from exc

        if proc.stdout is None:
            raise ProviderError("Codex process has no stdout")

        raw_fallback_lines: list[str] = []
        yielded_content = False

        async for raw_line in proc.stdout:
            line = raw_line.decode(errors="replace").strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                raw_fallback_lines.append(line)
                logger.debug("Non-JSON line from Codex: %s", line[:100])
                continue

            data: dict[str, Any] = event.get("data") or {}
            event_type = event.get("event") or event.get("type")

            # Extract assistant message content
            message = (
                data.get("assistant_message")
                or event.get("assistant_message")
                or data.get("message")
            )
            if isinstance(message, str) and message.strip():
                yield StreamChunk(content=message.strip() + "\n")
                yielded_content = True

            # Yield tool calls as they happen
            if event_type == "tool.call":
                tool_name = data.get("name") or data.get("tool")
                tool_args = _coerce_tool_args(data)
                if tool_name:
                    yield StreamChunk(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id=data.get("id", ""),
                                name=tool_name,
                                arguments=tool_args,
                            )
                        ],
                    )
                    yielded_content = True

            # Check for completion
            if event_type in ("turn.completed", "session.completed"):
                yield StreamChunk(content="", finish_reason="stop")

        await proc.wait()

        # Fallback if no structured content was yielded
        if not yielded_content and raw_fallback_lines:
            logger.warning("Codex output contained no parseable JSON events; using raw output")
            yield StreamChunk(content="\n".join(raw_fallback_lines))

        # Final chunk if we didn't get a completion event
        if proc.returncode != 0:
            yield StreamChunk(content="", finish_reason="error")


def is_codex_available() -> bool:
    """Check if Codex CLI is available in PATH."""
    return _find_codex_binary() is not None


__all__ = [
    "CODEX_AVAILABLE_MODELS",
    "CODEX_CONTEXT_LIMITS",
    "CodexNotFoundError",
    "CodexProvider",
    "is_codex_available",
]
