## Project Context

`jpscripts` is a modern, typed Python 3.11+ CLI toolbox. It replaces legacy Bash scripts with a Python architecture using **Typer** (CLI), **Rich** (UI), **Pydantic** (Configuration), and **GitPython**.

**Crucial:** Do not generate Bash scripts. Do not place code in `legacy/`. All new functionality belongs in `src/jpscripts/`.

## Code Style & Standards

### 1. File Headers & Imports

- **Must** include `from __future__ import annotations` as the first line in every Python file.
- Use absolute imports for internal modules (e.g., `from jpscripts.core.console import console`).
- Group imports: standard library, third-party, then local `jpscripts` modules.

### 2. Output & Logging (Strict)

- **NEVER** use Python's built-in `print()`.
- **ALWAYS** use the singleton `console` imported from `jpscripts.core.console`.
  - `console.print("[green]Success[/green]")` for user output.
  - `console.print(Table(...))` for structured data.
- For debugging or invisible logs, use the logger provided in `AppState`.
  - `state.logger.debug("...")`

### 3. Command Architecture (Typer)

All commands must follow this pattern to ensure they handle configuration (AppState) correctly:

```python
from __future__ import annotations
import typer
from jpscripts.core.console import console
from jpscripts.core.config import AppConfig

# 1. Function signature includes ctx: typer.Context
def my_command(ctx: typer.Context, name: str = typer.Argument(...)):
    """Docstring becomes the CLI help text."""

    # 2. Retrieve State
    state = ctx.obj  # type: jpscripts.main.AppState
    config: AppConfig = state.config

    # 3. Implementation
    # Use config.worktree_root, config.notes_dir, etc.

    # 4. Error Handling
    if some_error:
        console.print("[red]Error message[/red]")
        raise typer.Exit(code=1)
```

### 4. Configuration

- Configuration is strictly typed via Pydantic in `src/jpscripts/core/config.py`.
- Do not read `~/.jpconfig` manually. Rely on `state.config` passed via the context.
- If a new configuration field is needed, add it to the `AppConfig` class in `core/config.py` with a default factory or value.

### 5. Git Operations

- Prefer `jpscripts.core.git` helpers (`open_repo`, `describe_status`) over raw `git` commands.
- If high-performance async is needed (like scanning 100+ repos), follow the pattern in `src/jpscripts/commands/git_ops.py` using `asyncio` and `rich.Live`.

## Adding New Commands

1.  **Create Module:** Add `src/jpscripts/commands/<topic>.py`.
2.  **Register:** Import the function in `src/jpscripts/main.py` and register it:
    ```python
    app.command("command-name")(module.function_name)
    ```
3.  **External Tools:** If the command relies on a binary (e.g., `fzf`, `rg`), use `shutil.which` to validate presence. Raise `typer.Exit(1)` with a clear error if missing.

## Testing Standards

- Use `pytest`.
- **Fixtures:** You rely on `tests/conftest.py`.
  - `runner`: The `CliRunner` for invoking commands.
  - `isolate_config`: Ensures you don't overwrite the user's real `~/.jpconfig`.
  - `capture_console`: Mocks the Rich console so you can assert output.

**Example Test:**

```python
from typer.testing import CliRunner
import jpscripts.main as jp_main

def test_my_command(runner: CliRunner):
    result = runner.invoke(jp_main.app, ["command-name", "--flag"])
    assert result.exit_code == 0
    assert "Expected Output" in result.stdout
```

## Security & Safety

- **Sandbox:** Assume operations run in a user's actual shell environment.
- **Destructive Actions:** Ask for confirmation using `rich.prompt.Confirm.ask("Are you sure?")` for any deletion or irreversible git operation.
