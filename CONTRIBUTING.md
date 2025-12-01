# Contributing to jpscripts

## Architecture

This is a **Typer** application organized by domain:

- `src/jpscripts/main.py`: CLI bootstrap and dynamic command discovery/registration.
- `src/jpscripts/commands/`: Command modules (e.g., `nav.py`, `agent.py`).
- `src/jpscripts/core/`: Shared logic (config, console, runtime, engine).
  - `core/agent/`: Agent orchestration, prompting, execution, and repair strategies.
- `src/jpscripts/git/`: Git operations (status, diff, worktree management).
- `src/jpscripts/mcp/tools/`: MCP tool implementations, auto-registered by the MCP server.

## How to add a new command

1. **Create the module:** Add a new file in `src/jpscripts/commands/` (or extend an existing one) and define either:
   - `app = typer.Typer()` with subcommands, **or**
   - functions that match the dynamic registry patterns.
2. **Async I/O only:** Prefer `async` functions; wrap blocking calls in threads. Use `run_safe_shell` from `jpscripts.core.shell` for subprocess calls (it validates commands via `validate_command` and prevents shell injection).
3. **Document it:** Add the command to `README.md`.
4. **Test it:** Add focused unit tests in `tests/unit/` and a smoke entry in `tests/test_smoke.py` covering `--help` and a basic invocation.

## How to add a new MCP Tool

1. **Create the module:** Add a new file in `src/jpscripts/mcp/tools/`.
2. **Define the tool:** Write an `async` function and decorate it with `@tool` and `@tool_error_handler`.
3. **Validate paths:** Use `security.validate_path(...)` for any filesystem access.
4. **Auto-registration:** Tools are discovered and registered automatically on the next run—no manual wiring.
5. **Test it:** Add unit tests in `tests/unit/` (use `pytest-asyncio` for awaitables).

## Definition of Done (checklist)

- [ ] Unit tests added/updated (`pytest-asyncio` for async paths; smoke test for new CLI commands).
- [ ] Full type hints on public surfaces (Pydantic models validated where applicable).
- [ ] No blocking I/O: uses `asyncio` or wraps blocking calls in threads; no `shell=True`.

## Development

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev,ai]"

# Run tests
pytest
```
