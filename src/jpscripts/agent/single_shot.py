"""Single-shot agent execution without repair loop.

This module provides a simplified interface for running a single agent
prompt without the full repair loop machinery. Useful for one-off queries
or when you need direct control over the agent interaction.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError

from jpscripts.agent.models import AgentResponse, PreparedPrompt, ResponseFetcher
from jpscripts.agent.parsing import parse_agent_response
from jpscripts.agent.prompting import prepare_agent_prompt


@dataclass
class SingleShotConfig:
    """Configuration for single-shot agent execution.

    Attributes:
        attach_recent: Whether to attach recent file changes as context.
        include_diff: Whether to include git diff in the context.
        web_access: Whether to enable web search capabilities.
    """

    attach_recent: bool = False
    include_diff: bool = True
    web_access: bool = False


@dataclass
class SingleShotResult:
    """Result of single-shot agent execution.

    Attributes:
        raw_response: The raw string response from the LLM.
        prepared: The prepared prompt that was sent.
        agent_response: Parsed agent response if successful.
        error: Error message if parsing failed.
    """

    raw_response: str
    prepared: PreparedPrompt
    agent_response: AgentResponse | None = None
    error: str | None = None


class SingleShotRunner:
    """Runs a single-shot agent prompt without a repair loop.

    This class encapsulates the logic for preparing a prompt, fetching
    a response from an LLM, and parsing the result. It enables programmatic
    invocation of the agent without CLI dependencies.

    Example:
        runner = SingleShotRunner(
            prompt="Explain this code",
            model="claude-sonnet-4",
            fetch_response=my_fetcher,
            config=SingleShotConfig(attach_recent=True),
        )
        result = await runner.run()
        if result.agent_response:
            print(result.agent_response.thought_process)
    """

    def __init__(
        self,
        *,
        prompt: str,
        model: str,
        fetch_response: ResponseFetcher,
        config: SingleShotConfig | None = None,
        run_command: str | None = None,
    ) -> None:
        """Initialize the single-shot runner.

        Args:
            prompt: The user's instruction/prompt.
            model: LLM model ID to use.
            fetch_response: Async function to fetch LLM responses.
            config: Configuration options.
            run_command: Optional command to run for context gathering.
        """
        self.prompt = prompt
        self.model = model
        self.fetch_response = fetch_response
        self.config = config or SingleShotConfig()
        self.run_command = run_command

    async def prepare(self) -> PreparedPrompt:
        """Prepare the prompt with context.

        Returns:
            PreparedPrompt ready to send to the LLM.
        """
        return await prepare_agent_prompt(
            base_prompt=self.prompt,
            model=self.model,
            run_command=self.run_command,
            attach_recent=self.config.attach_recent,
            include_diff=self.config.include_diff,
            web_access=self.config.web_access,
        )

    async def run(self) -> SingleShotResult:
        """Execute the single-shot agent call.

        Returns:
            SingleShotResult with the response and parsed agent response.
        """
        prepared = await self.prepare()
        raw_response = await self.fetch_response(prepared)

        if not raw_response:
            return SingleShotResult(
                raw_response="",
                prepared=prepared,
                error="No response received from agent.",
            )

        try:
            agent_response = parse_agent_response(raw_response)
            return SingleShotResult(
                raw_response=raw_response,
                prepared=prepared,
                agent_response=agent_response,
            )
        except ValidationError as exc:
            return SingleShotResult(
                raw_response=raw_response,
                prepared=prepared,
                error=f"Agent response validation failed: {exc}",
            )


__all__ = [
    "SingleShotConfig",
    "SingleShotResult",
    "SingleShotRunner",
]
