# Extending jpscripts

This guide explains how to extend jpscripts with custom commands, MCP tools, agents, and providers.

---

## Table of Contents

1. [Adding a CLI Command](#adding-a-cli-command)
2. [Adding an MCP Tool](#adding-an-mcp-tool)
3. [Adding an Agent Persona](#adding-an-agent-persona)
4. [Adding an LLM Provider](#adding-an-llm-provider)
5. [Adding Memory Patterns](#adding-memory-patterns)
6. [Testing Extensions](#testing-extensions)

---

## Adding a CLI Command

### Basic Command

Create a new file in `src/jpscripts/commands/`:

```python
# src/jpscripts/commands/mycommand.py
"""My custom command."""

import typer

app = typer.Typer(help="My custom command group")


@app.command()
def hello(
    name: str = typer.Option("World", "--name", "-n", help="Name to greet"),
    loud: bool = typer.Option(False, "--loud", "-l", help="Shout the greeting"),
) -> None:
    """Say hello to someone."""
    message = f"Hello, {name}!"
    if loud:
        message = message.upper()
    typer.echo(message)


@app.command()
def goodbye(name: str = typer.Argument("World", help="Name to say goodbye to")) -> None:
    """Say goodbye to someone."""
    typer.echo(f"Goodbye, {name}!")
```

The command is automatically discovered and registered. Run it with:

```bash
jp mycommand hello --name Alice
jp mycommand goodbye Bob
```

### Async Command

For commands that perform I/O:

```python
# src/jpscripts/commands/fetch.py
"""Fetch data from remote sources."""

import asyncio
from pathlib import Path

import typer

from jpscripts.core.console import console
from jpscripts.core.shell import run_safe_shell

app = typer.Typer(help="Fetch operations")


@app.command()
def url(
    target: str = typer.Argument(..., help="URL to fetch"),
    output: Path = typer.Option(None, "--output", "-o", help="Output file"),
) -> None:
    """Fetch a URL and display or save the content."""
    result = asyncio.run(_fetch_url(target, output))
    if result:
        console.print(result)


async def _fetch_url(url: str, output: Path | None) -> str:
    """Async implementation of URL fetching."""
    # Use safe shell for curl
    result = await run_safe_shell(["curl", "-sL", url])

    if output:
        output.write_text(result.stdout)
        return f"Saved to {output}"

    return result.stdout
```

### Command with Subcommands

```python
# src/jpscripts/commands/db.py
"""Database operations."""

import typer

app = typer.Typer(help="Database commands")

migrate_app = typer.Typer(help="Migration commands")
app.add_typer(migrate_app, name="migrate")


@migrate_app.command("up")
def migrate_up(steps: int = typer.Option(1, help="Number of migrations")) -> None:
    """Run pending migrations."""
    typer.echo(f"Running {steps} migration(s) up...")


@migrate_app.command("down")
def migrate_down(steps: int = typer.Option(1, help="Number of migrations")) -> None:
    """Rollback migrations."""
    typer.echo(f"Rolling back {steps} migration(s)...")
```

Usage:
```bash
jp db migrate up --steps 3
jp db migrate down
```

---

## Adding an MCP Tool

MCP (Model Context Protocol) tools allow LLMs to interact with your system.

### Basic Tool

Create a new file in `src/jpscripts/mcp/tools/`:

```python
# src/jpscripts/mcp/tools/weather.py
"""Weather information tools."""

from mcp.server.fastmcp import tool


@tool
async def get_weather(city: str) -> str:
    """Get current weather for a city.

    Args:
        city: Name of the city

    Returns:
        Weather description
    """
    # In a real implementation, call a weather API
    return f"The weather in {city} is sunny, 22°C"


@tool
async def get_forecast(city: str, days: int = 3) -> str:
    """Get weather forecast for a city.

    Args:
        city: Name of the city
        days: Number of days to forecast (1-7)

    Returns:
        Forecast description
    """
    if days < 1 or days > 7:
        return "Error: days must be between 1 and 7"

    return f"{days}-day forecast for {city}: Sunny → Cloudy → Rain"
```

### Tool with File Access

For tools that access the filesystem, always validate paths:

```python
# src/jpscripts/mcp/tools/analysis.py
"""Code analysis tools."""

from pathlib import Path

from mcp.server.fastmcp import tool

from jpscripts.core import security


@tool
async def analyze_file(path: str) -> str:
    """Analyze a source file and return metrics.

    Args:
        path: Path to the file to analyze

    Returns:
        Analysis results
    """
    # ALWAYS validate paths
    validated = security.validate_path(path)

    if not validated.exists():
        return f"Error: File not found: {path}"

    content = validated.read_text()
    lines = len(content.splitlines())
    chars = len(content)

    return f"File: {path}\nLines: {lines}\nCharacters: {chars}"


@tool
async def search_code(pattern: str, directory: str = ".") -> str:
    """Search for a pattern in code files.

    Args:
        pattern: Regex pattern to search for
        directory: Directory to search in

    Returns:
        Matching files and lines
    """
    validated_dir = security.validate_path(directory)

    if not validated_dir.is_dir():
        return f"Error: Not a directory: {directory}"

    # Implementation...
    return "Search results..."
```

### Tool with Rate Limiting

Add rate limiting for resource-intensive tools:

```python
# src/jpscripts/mcp/tools/expensive.py
"""Resource-intensive tools."""

from mcp.server.fastmcp import tool

from jpscripts.core.rate_limit import RateLimiter

# 10 operations per minute
_limiter = RateLimiter(tokens_per_second=10/60, burst=5)


@tool
async def expensive_operation(data: str) -> str:
    """Perform an expensive operation.

    Args:
        data: Input data

    Returns:
        Operation result
    """
    if not _limiter.try_acquire():
        return "Error: Rate limit exceeded. Please wait before retrying."

    # Expensive operation...
    return f"Processed: {data}"
```

---

## Adding an Agent Persona

Agent personas define specialized behaviors for the AI agent.

### Create a Persona File

Add to `AGENTS.md` or create a persona configuration:

```markdown
## Analyst

**Role**: Data analysis and visualization specialist

**Strengths**:
- Statistical analysis
- Data visualization
- Pattern recognition
- Report generation

**Approach**:
1. Understand the data structure
2. Identify key metrics and patterns
3. Create visualizations
4. Summarize findings

**Tools preferred**:
- `analyze_data`
- `create_chart`
- `export_report`

**Constraints**:
- Always validate data before analysis
- Use appropriate statistical methods
- Explain findings in plain language
```

### Implement Persona Logic

In `src/jpscripts/core/agent/`:

```python
# src/jpscripts/core/agent/personas.py
"""Agent persona implementations."""

from dataclasses import dataclass
from typing import Literal

PersonaType = Literal["analyst", "developer", "reviewer", "architect"]


@dataclass
class Persona:
    """Agent persona configuration."""

    name: PersonaType
    system_prompt: str
    preferred_tools: list[str]
    temperature: float = 0.7


PERSONAS: dict[PersonaType, Persona] = {
    "analyst": Persona(
        name="analyst",
        system_prompt="""You are a data analyst. Your approach:
1. Examine data structure and quality
2. Identify patterns and anomalies
3. Create clear visualizations
4. Provide actionable insights""",
        preferred_tools=["analyze_file", "search_code"],
        temperature=0.3,  # More deterministic
    ),
    "developer": Persona(
        name="developer",
        system_prompt="""You are a software developer. Your approach:
1. Understand requirements
2. Write clean, tested code
3. Follow project conventions
4. Document your changes""",
        preferred_tools=["read_file", "write_file", "run_command"],
        temperature=0.7,
    ),
}


def get_persona(name: PersonaType) -> Persona:
    """Get a persona by name."""
    return PERSONAS[name]
```

---

## Adding an LLM Provider

### Provider Interface

Implement the Provider protocol:

```python
# src/jpscripts/providers/custom.py
"""Custom LLM provider implementation."""

from collections.abc import AsyncIterator
from typing import Any

from jpscripts.providers import (
    CompletionOptions,
    CompletionResponse,
    Message,
    Provider,
    ProviderError,
)


class CustomProvider(Provider):
    """Custom LLM provider."""

    def __init__(self, api_key: str, base_url: str = "https://api.example.com"):
        self.api_key = api_key
        self.base_url = base_url
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Lazy-load the HTTP client."""
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        return self._client

    async def complete(
        self,
        messages: list[Message],
        model: str,
        options: CompletionOptions | None = None,
    ) -> CompletionResponse:
        """Generate a completion."""
        opts = options or CompletionOptions()

        try:
            response = await self.client.post(
                "/v1/completions",
                json={
                    "model": model,
                    "messages": [{"role": m.role, "content": m.content} for m in messages],
                    "temperature": opts.temperature,
                    "max_tokens": opts.max_tokens,
                },
            )
            response.raise_for_status()
            data = response.json()

            return CompletionResponse(
                content=data["choices"][0]["message"]["content"],
                model=model,
                usage={
                    "prompt_tokens": data["usage"]["prompt_tokens"],
                    "completion_tokens": data["usage"]["completion_tokens"],
                },
            )

        except Exception as exc:
            raise ProviderError(f"Custom provider error: {exc}") from exc

    async def stream(
        self,
        messages: list[Message],
        model: str,
        options: CompletionOptions | None = None,
    ) -> AsyncIterator[str]:
        """Stream a completion."""
        # Implementation for streaming...
        yield "Streaming not implemented"

    def is_available(self) -> bool:
        """Check if the provider is available."""
        return bool(self.api_key)
```

### Register the Provider

Update `src/jpscripts/providers/factory.py`:

```python
def get_provider(config: AppConfig, model_id: str) -> Provider:
    """Get a provider for the given model."""
    if model_id.startswith("custom/"):
        from jpscripts.providers.custom import CustomProvider
        return CustomProvider(
            api_key=os.environ.get("CUSTOM_API_KEY", ""),
            base_url=config.custom_provider_url,
        )

    # ... existing provider logic
```

---

## Adding Memory Patterns

Memory patterns allow the agent to learn and recall information.

### Pattern Definition

```python
# src/jpscripts/core/memory/patterns.py

from dataclasses import dataclass
from typing import Literal

PatternType = Literal["error", "success", "preference", "fact"]


@dataclass
class Pattern:
    """A learned pattern."""

    type: PatternType
    trigger: str  # What triggers this pattern
    response: str  # How to respond
    confidence: float = 0.5
    occurrences: int = 1


# Example patterns
BUILTIN_PATTERNS = [
    Pattern(
        type="error",
        trigger="ImportError: No module named",
        response="Check if the module is installed: pip install <module>",
        confidence=0.9,
    ),
    Pattern(
        type="success",
        trigger="All tests passed",
        response="Tests are green, safe to commit",
        confidence=0.95,
    ),
]
```

### Pattern Storage

```python
# Add to memory store

async def save_pattern(pattern: Pattern) -> None:
    """Save a pattern to memory."""
    entry = MemoryEntry(
        id=generate_id(),
        ts=now_iso(),
        content=f"Pattern: {pattern.trigger} → {pattern.response}",
        tags=["pattern", pattern.type],
        metadata={"confidence": pattern.confidence},
    )
    await save_memory(entry)


async def find_patterns(context: str) -> list[Pattern]:
    """Find patterns matching the context."""
    entries = await query_memory(context, limit=5, tags=["pattern"])
    return [entry_to_pattern(e) for e in entries]
```

---

## Testing Extensions

### Testing CLI Commands

```python
# tests/unit/test_mycommand.py
import pytest
from typer.testing import CliRunner

from jpscripts.commands.mycommand import app

runner = CliRunner()


def test_hello_default():
    result = runner.invoke(app, ["hello"])
    assert result.exit_code == 0
    assert "Hello, World!" in result.output


def test_hello_with_name():
    result = runner.invoke(app, ["hello", "--name", "Alice"])
    assert result.exit_code == 0
    assert "Hello, Alice!" in result.output


def test_hello_loud():
    result = runner.invoke(app, ["hello", "--loud"])
    assert result.exit_code == 0
    assert "HELLO, WORLD!" in result.output
```

### Testing MCP Tools

```python
# tests/unit/test_weather_tool.py
import pytest

from jpscripts.mcp.tools.weather import get_weather, get_forecast


@pytest.mark.asyncio
async def test_get_weather():
    result = await get_weather("London")
    assert "London" in result
    assert "sunny" in result.lower() or "weather" in result.lower()


@pytest.mark.asyncio
async def test_get_forecast_valid():
    result = await get_forecast("Paris", days=5)
    assert "forecast" in result.lower()


@pytest.mark.asyncio
async def test_get_forecast_invalid_days():
    result = await get_forecast("Tokyo", days=10)
    assert "Error" in result
```

### Testing Providers

```python
# tests/unit/test_custom_provider.py
import pytest
from unittest.mock import AsyncMock, patch

from jpscripts.providers.custom import CustomProvider
from jpscripts.providers import Message, CompletionOptions


@pytest.fixture
def provider():
    return CustomProvider(api_key="test-key")


@pytest.mark.asyncio
async def test_complete_success(provider):
    with patch.object(provider, "client") as mock_client:
        mock_client.post = AsyncMock(return_value=MockResponse({
            "choices": [{"message": {"content": "Hello!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }))

        result = await provider.complete(
            messages=[Message(role="user", content="Hi")],
            model="custom-model",
        )

        assert result.content == "Hello!"
        assert result.usage["prompt_tokens"] == 10
```

### Adding Smoke Tests

```python
# tests/test_smoke.py

def test_mycommand_help(runner):
    """Test mycommand --help works."""
    from jpscripts.main import app
    result = runner.invoke(app, ["mycommand", "--help"])
    assert result.exit_code == 0
    assert "My custom command" in result.output
```

---

## Best Practices

### Security

1. **Always validate paths** with `security.validate_path()`
2. **Never use `shell=True`** in subprocess calls
3. **Redact secrets** from error messages
4. **Rate limit** resource-intensive tools

### Performance

1. **Lazy load** heavy dependencies
2. **Use async I/O** for network/file operations
3. **Cache** expensive computations
4. **Stream** large responses

### Compatibility

1. **Follow existing patterns** in the codebase
2. **Add type hints** to all public APIs
3. **Write tests** for new functionality
4. **Document** user-facing features

---

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Development guide
- [HANDBOOK.md](../HANDBOOK.md) - Agent protocol
