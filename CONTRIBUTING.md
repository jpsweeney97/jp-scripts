# Contributing to jpscripts

Welcome! This guide covers everything you need to contribute effectively.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Development Setup](#development-setup)
3. [Testing](#testing)
4. [Git Workflow](#git-workflow)
5. [Debugging](#debugging)
6. [How to Add Features](#how-to-add-features)
7. [Release Process](#release-process)

---

## Architecture

This is a **Typer** application organized by domain:

```
src/jpscripts/
├── main.py              # CLI bootstrap and dynamic command discovery
├── commands/            # Command modules (nav.py, agent.py, etc.)
├── core/                # Shared logic
│   ├── agent/           # Agent orchestration, prompting, execution
│   ├── engine/          # LLM provider integration, response parsing
│   ├── memory/          # Memory storage, retrieval, embeddings
│   ├── config.py        # Configuration management
│   ├── console.py       # Rich console utilities
│   ├── governance.py    # Constitutional AI safety checks
│   ├── security.py      # Path validation, TOCTOU-safe operations
│   ├── shell.py         # Safe subprocess execution
│   └── runtime.py       # Runtime context variables
├── git/                 # Git operations (status, diff, worktree)
├── mcp/                 # MCP server and tools
│   └── tools/           # Auto-registered MCP tool implementations
└── providers/           # LLM provider implementations (anthropic, openai)
```

### Layer Boundaries

- **CLI commands** (`commands/`) may import from: `core/`, `git/`, `providers/`
- **CLI commands must NOT** import from `mcp/` — move shared logic to `core/`
- **MCP tools** (`mcp/tools/`) may import from: `core/`, `git/`
- **Providers** receive config values via parameters, not direct `AppConfig` imports

---

## Development Setup

```bash
# Clone and install in editable mode
git clone <repo-url>
cd jp-scripts
pip install -e ".[dev,ai]"

# Verify installation
jp --help
make test
```

### Optional Dependencies

```bash
# AI features (embeddings, LanceDB)
pip install -e ".[ai]"

# LLM providers
pip install -e ".[providers]"

# OpenTelemetry tracing
pip install -e ".[otel]"

# All dependencies
pip install -e ".[dev,ai,providers,otel]"
```

---

## Testing

### Running Tests

```bash
# Full test suite with linting and coverage
make test

# Fast test run (skip linting)
pytest

# Run with coverage report
pytest --cov=src/jpscripts --cov-report=term-missing

# Run specific test file
pytest tests/unit/test_memory.py

# Run specific test
pytest tests/unit/test_memory.py::test_score_function -v

# Run only fast tests (skip slow/integration)
pytest -m "not slow"

# Run only integration tests
pytest tests/integration/
```

### Test Directory Structure

```
tests/
├── conftest.py          # Shared fixtures (runner, isolate_config, etc.)
├── test_smoke.py        # CLI smoke tests for all commands
├── unit/                # Fast, isolated unit tests
├── integration/         # Tests that hit real services or subprocess
├── security/            # Security-focused tests
├── properties/          # Property-based tests (Hypothesis)
└── mocks/               # Reusable mock implementations
```

### Pytest Markers

```python
@pytest.mark.slow  # Tests that take >1s or hit external services
```

Run without slow tests: `pytest -m "not slow"`

### Key Fixtures (from conftest.py)

- **`runner`**: `CliRunner` for testing CLI commands
- **`isolate_config`**: Redirects config to temp dir (autouse)
- **`capture_console`**: In-memory Rich console (autouse)

### Coverage Expectations

- **Target**: 70%+ overall coverage
- **Critical modules**: 80%+ coverage required for:
  - `core/error_middleware.py`
  - `core/rate_limit.py`
  - `core/security.py`
  - `core/governance.py`

### Writing Tests

```python
# Unit test example
import pytest
from jpscripts.memory import MemoryEntry

def test_memory_entry_validation():
    entry = MemoryEntry(id="123", ts="2024-01-01", content="test", tags=["foo"])
    assert entry.id == "123"
    assert "foo" in entry.tags


# Async test example
import pytest

@pytest.mark.asyncio
async def test_async_search():
    result = await some_async_function()
    assert result is not None


# Property-based test example (Hypothesis)
from hypothesis import given, strategies as st

@given(st.text(min_size=1))
def test_tokenize_never_crashes(text: str):
    tokens = _tokenize(text)
    assert isinstance(tokens, list)
```

### Test Best Practices

1. **Test failure paths**, not just happy paths
2. **Avoid over-mocking** — test actual behavior where possible
3. **Use fixtures** for common setup (see conftest.py)
4. **Mark slow tests** with `@pytest.mark.slow`
5. **Safety-critical code must have tests** (error handling, security, rate limiting)

---

## Git Workflow

### Branch Naming

```
feature/add-memory-search
fix/symlink-escape-vulnerability
refactor/split-engine-module
docs/update-contributing
```

### Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code restructuring (no behavior change)
- `docs`: Documentation only
- `test`: Adding/updating tests
- `perf`: Performance improvement
- `ci`: CI/CD changes
- `security`: Security fix
- `chore`: Maintenance tasks

**Examples:**
```
feat(memory): add semantic search with embeddings
fix(governance): prevent path traversal in patch application
refactor(engine): extract response parsing to separate module
test(providers): add error handling coverage for Anthropic
```

### Pull Request Process

1. **Create a feature branch** from `main`
2. **Make focused commits** — each commit should be atomic and pass tests
3. **Run tests locally** before pushing: `make test`
4. **Push and create PR** using the template
5. **Address review feedback** with additional commits
6. **Squash or rebase** if requested before merge

### PR Checklist (from template)

- [ ] Tests pass locally (`make test`)
- [ ] Linting passes (`make lint`)
- [ ] New tests added for new functionality
- [ ] No secrets or credentials included
- [ ] Breaking changes documented

---

## Debugging

### Common Issues

#### Import Errors

```bash
# Ensure editable install is up to date
pip install -e ".[dev,ai]"

# Check for circular imports
python -c "from jpscripts.core import memory"
```

#### Test Failures

```bash
# Run with verbose output
pytest tests/unit/test_memory.py -v

# Run with stdout capture disabled (see print statements)
pytest -s

# Run with pdb on failure
pytest --pdb

# Run specific failing test
pytest tests/unit/test_memory.py::test_failing_case -v
```

#### Type Errors

```bash
# Run mypy with verbose output
mypy src --show-error-codes

# Check specific file
mypy src/jpscripts/core/memory/store.py
```

### Logging

Enable debug logging for development:

```python
from jpscripts.core.console import get_logger

logger = get_logger(__name__)
logger.debug("Detailed diagnostic info")
logger.warning("Potential issue: %s", details)
```

### Tracing

For agent debugging, enable OpenTelemetry tracing:

```bash
# Install tracing dependencies
pip install -e ".[otel]"

# Run with tracing enabled
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 jp agent run
```

### Interactive Debugging

```python
# Add breakpoint in code
import pdb; pdb.set_trace()

# Or use the built-in
breakpoint()

# Run test with debugger
pytest --pdb tests/unit/test_memory.py::test_specific
```

---

## How to Add Features

### Adding a New CLI Command

1. **Create the module** in `src/jpscripts/commands/`:
   ```python
   # commands/mycommand.py
   import typer
   app = typer.Typer()

   @app.command()
   def run(name: str = typer.Option(..., help="Name to greet")):
       """My command description."""
       typer.echo(f"Hello, {name}!")
   ```

2. **Use async for I/O** — wrap blocking calls:
   ```python
   import asyncio
   from jpscripts.core.shell import run_safe_shell

   @app.command()
   def run():
       result = asyncio.run(_async_run())
       typer.echo(result)

   async def _async_run():
       return await run_safe_shell(["ls", "-la"])
   ```

3. **Add smoke test** in `tests/test_smoke.py`:
   ```python
   def test_mycommand_help(runner):
       result = runner.invoke(app, ["mycommand", "--help"])
       assert result.exit_code == 0
   ```

4. **Document** in README.md

### Adding a New MCP Tool

1. **Create the module** in `src/jpscripts/mcp/tools/`:
   ```python
   # mcp/tools/mytool.py
   from mcp.server.fastmcp import tool
   from jpscripts.core import security

   @tool
   async def my_tool(path: str) -> str:
       """Tool description for the LLM."""
       validated = security.validate_path(path)
       # Implementation...
       return result
   ```

2. **Validate all paths** — use `security.validate_path()` or `security.validate_path_safe()`

3. **Add tests** in `tests/unit/test_mcp_*.py`

4. **Auto-registration** — tools are discovered automatically, no manual wiring needed

### Definition of Done

- [ ] Unit tests added/updated (`pytest-asyncio` for async)
- [ ] Smoke test for new CLI commands
- [ ] Full type hints on public surfaces
- [ ] `mypy` passes with no errors
- [ ] No blocking I/O (uses `asyncio` or threads)
- [ ] No `shell=True` in subprocess calls
- [ ] Paths validated for filesystem access
- [ ] Documentation updated if user-facing

---

## Error Handling Patterns

This codebase uses two complementary error handling patterns. Use the right one for the situation.

### Result[T, E] Pattern

Use `Result[T, E]` (from `jpscripts.core.result`) for **expected, recoverable errors**:

```python
from jpscripts.core.result import Result, Ok, Err, ConfigurationError

def load_config(path: Path) -> Result[AppConfig, ConfigurationError]:
    """Load configuration from file.

    Returns Ok(config) on success, Err(error) on failure.
    """
    if not path.exists():
        return Err(ConfigurationError(f"Config file not found: {path}"))

    try:
        data = path.read_text()
        config = parse_config(data)
        return Ok(config)
    except ValueError as exc:
        return Err(ConfigurationError(f"Invalid config format: {exc}"))
```

**When to use Result:**
- File I/O that might fail (missing files, permissions)
- Network requests (timeouts, connection errors)
- Parsing user input (invalid format)
- Configuration loading
- Any operation where failure is expected and recoverable

**Consuming Results with match:**
```python
match load_config(path):
    case Ok(config):
        use_config(config)
    case Err(error):
        logger.error("Config error: %s", error)
        sys.exit(1)
```

### Exceptions

Use exceptions for **unexpected errors and programming bugs**:

```python
def process_item(item: Item) -> None:
    """Process a single item.

    Raises:
        ValueError: If item is in invalid state (programming error)
    """
    if item.status not in VALID_STATUSES:
        raise ValueError(f"Invalid item status: {item.status}")

    # Processing logic...
```

**When to use exceptions:**
- Programming errors (invalid arguments, logic bugs)
- Invariant violations that "should never happen"
- Deep in call stacks where propagation is cleaner
- Third-party library errors (let them bubble up)

### Guidelines

| Scenario | Pattern | Example |
|----------|---------|---------|
| File not found | `Result` | User provided invalid path |
| Network timeout | `Result` | API call failed |
| Invalid user input | `Result` | Malformed JSON in request |
| Null/None where forbidden | Exception | Programming bug |
| Invalid state transition | Exception | Logic error |
| Third-party lib error | Exception | Let it propagate |

### Exception Variable Naming

Always use `exc` as the exception variable name (not `e` or `err`):

```python
# Good
try:
    something()
except ValueError as exc:
    logger.warning("Operation failed: %s", exc)

# Bad
try:
    something()
except ValueError as e:  # Don't use 'e'
    ...
```

### Never Swallow Errors Silently

```python
# Bad - silent failure
try:
    risky_operation()
except Exception:
    pass  # Never do this!

# Good - log and handle
try:
    risky_operation()
except Exception as exc:
    logger.warning("Operation failed: %s", exc)
    return fallback_value
```

---

## Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (1.0.0): Breaking API changes
- **MINOR** (0.1.0): New features, backwards compatible
- **PATCH** (0.0.1): Bug fixes, backwards compatible

Current version is in `pyproject.toml`:
```toml
[project]
version = "0.9.0"
```

### Release Checklist

1. **Ensure all tests pass**:
   ```bash
   make test
   ```

2. **Update version** in `pyproject.toml`

3. **Update CHANGELOG** (if maintained)

4. **Create release commit**:
   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "chore: release v0.9.1"
   ```

5. **Tag the release**:
   ```bash
   git tag v0.9.1
   git push origin main --tags
   ```

6. **Build and publish** (if applicable):
   ```bash
   pip install build twine
   python -m build
   twine upload dist/*
   ```

### Breaking Changes

When making breaking changes:

1. **Document** in PR description and CHANGELOG
2. **Bump MAJOR version** (or MINOR if pre-1.0)
3. **Provide migration guide** if the change affects users

---

## Questions?

Open an issue or check existing discussions. We're happy to help!
