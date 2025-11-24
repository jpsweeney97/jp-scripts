## 1. Project Philosophy & Architecture

`jpscripts` is a high-leverage, "God-Mode" CLI for macOS power users. It orchestrates system binaries (`git`, `fzf`, `rg`, `gh`) into unified, typed Python workflows.

### Core Principles

- **Speed is feature #1:** Use `asyncio` for multi-repo scanning (see `git_ops.py`).
- **Rich UI:** All output must use `rich`. No raw print statements.
- **Fail Gracefully:** We wrap subprocess calls. We do not crash with raw Python tracebacks unless `--verbose` is set.
- **Typed & Configured:** All state passes through `AppState` (Typer context) and is typed via Pydantic (`AppConfig`).

### Architectural Map

- **`src/jpscripts/main.py`**: The entry point. Handles `AppState` injection, logging setup, and the `ExternalTool` registry.
- **`src/jpscripts/core/`**: Shared libraries.
  - `console.py`: The singleton `console` instance.
  - `config.py`: Pydantic models handling the TOML/Env var/CLI flag precedence.
  - `git.py`: `GitPython` wrappers. **Always** use these over raw git commands when possible.
- **`src/jpscripts/commands/`**: Domain-specific logic. Each file is a module grouping related commands (e.g., `git_ops`, `nav`).

## 2. Implementation Standards

### Command Structure (Strict)

Every command **must** accept `ctx: typer.Context` to access global configuration.

```python
@app.command()
def my_command(ctx: typer.Context, arg: str):
    state: AppState = ctx.obj
    # usage: state.config.notes_dir, state.logger.debug()
```

### Subprocess & External Tools

We prefer `asyncio.create_subprocess_exec` for parallel tasks (see `doctor` command) and `subprocess.run` for blocking tasks.

- **Validation:** ALWAYS check `shutil.which("binary")` before execution.
- **FZF Integration:** Use the pattern in `system.py` or `nav.py`. Gracefully fallback to `Table` output if `fzf` is missing or `--no-fzf` is passed.

### Output & Logging

- **User Output:** `console.print()`
- **Debug/Internal:** `state.logger.debug()`
- **Errors:**
  ```python
  console.print(f"[red]Specific error message[/red]")
  raise typer.Exit(code=1)
  ```

## 3. Review Guidelines

**@codex review**: When reviewing PRs or diffs, enforce the following priorities:

- **P0 (Critical):**
  - usage of `print()` instead of `console.print()`.
  - usage of `sys.exit()` instead of `typer.Exit()`.
  - hardcoded paths (must use `pathlib` and `state.config`).
  - missing `shutil.which()` checks before calling external binaries.
  - Blocking I/O inside `async` functions (file reads/writes).
- **P1 (Important):**
  - Missing type hints (`from __future__ import annotations` is required).
  - Lack of user feedback (e.g., silent success).
  - Imports not grouped (Stdlib -\> Third Party -\> Local).
- **P2 (Nice to have):**
  - Docstrings for every command (these appear in `--help`).

## 4. Testing Strategy

We use `pytest` with `typer.testing.CliRunner`.

- **Smoke Tests:** Located in `tests/test_smoke.py`. Ensure the command runs and returns exit code 0.
- **Mocking:** Use `monkeypatch` to mock `shutil.which` or `subprocess.run` for commands that require external tools not present in the CI environment.
- **Config Isolation:** Always use the `isolate_config` fixture to prevent tests from reading the developer's actual `~/.jpconfig`.

## 5. Common Tasks

### Adding a Dependency

1.  Add to `pyproject.toml` under `dependencies`.
2.  If it is a CLI tool (e.g., `jq`), add it to `DEFAULT_TOOLS` in `main.py` so `jp doctor` checks for it.

### Adding a Config Option

1.  Add the field to `AppConfig` in `src/jpscripts/core/config.py`.
2.  Add the mapping in `ENV_VAR_MAP` if it should be overrideable via env vars.
3.  Update `init.py` to prompt for it during setup.

## Developer Constraints

1.  **Documentation First:** You are FORBIDDEN from adding a new command to `src/jpscripts/main.py` without simultaneously adding it to the "Command Reference" table in `README.md`.
2.  **Type Safety:** All new Python files must start with `from __future__ import annotations`.
3.  **UI Consistency:** Never use `print()`. Always use `console.print()` from `jpscripts.core.console`.
4.  **No Shell Scripts:** This is a Python project. Do not create `.sh` files.

## Tool Manifest (Architecture)

When modifying `jpscripts`, prefer these internal wrappers over raw subprocess calls:

- **Git:** `jpscripts.core.git` (Use `open_repo`, `describe_status`)
- **Config:** `jpscripts.core.config.AppConfig` (Available via `ctx.obj.config`)
- **Logging:** `jpscripts.core.console.setup_logging` (Rich-integrated logging)

## Testing

- **Runner:** `pytest`
- **Smoke Tests:** `tests/test_smoke.py` (Use this to verify CLI invocation)
- **Mocking:** Use `typer.testing.CliRunner` for all command tests.

## 6. Critical Invariants

- **Concurrency:** All file writes MUST be atomic or locked. No race conditions allowed in CLI tools that might run in parallel panes.
- **Platform Agnosticism:** Core logic must run on Linux/macOS. UI layers (like opening files) must detect the OS.
- **Configuration:** Never hardcode ignore lists or paths. All magic strings move to `AppConfig` or `DEFAULT_CONSTANTS` that can be overridden.
