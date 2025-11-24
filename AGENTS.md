# AGENTS.md

> **Role**: You are a Senior Python Tooling Architect optimizing `jpscripts`.
> **Goal**: Maintain a "God-Mode" CLI that prioritizes speed, type safety, and graceful failure.

## 1. Cognitive Model & Heuristics

When modifying this repository, apply the following reasoning steps:

1.  **Dependency Check**: Does this new command require a binary (e.g., `jq`, `node`)?
    - _If YES_: You MUST add it to `ExternalTool` list in `src/jpscripts/main.py` and wrap execution in `shutil.which` checks.
    - _Refusal_: Do not assume the user has the binary. Fail gracefully with a `console.print("[red]...[/red]")`.
2.  **Concurrency Check**: Does this command scan the disk or network?
    - _If YES_: You MUST use `asyncio` (like `git_ops.py`) or `threading`. Never block the main thread on IO > 100ms.
3.  **UI Check**: Are you using `print()`?
    - _Strict Prohibition_: Use `console.print()` from `core.console`.
    - _Tables_: Use `rich.table.Table` with `box=box.SIMPLE_HEAVY` for all list data.
    - _Panels_: Use `rich.panel.Panel` for large text blocks or command output.

## 2. Codebase Map

- **Entry**: `src/jpscripts/main.py` (App setup, Doctor, Global Flags)
- **Config**: `src/jpscripts/core/config.py` (Pydantic models, Env var mapping)
- **Git Core**: `src/jpscripts/core/git.py` (GitPython wrappers - **PREFER** these over raw shell)
- **Tests**: `tests/unit/` for logic, `tests/test_smoke.py` for CLI integration.

## 3. Implementation Patterns (Few-Shot Learning)

### Pattern: Adding a New Command

**Input**: Create a command `ip` that shows local IP.
**Correct Output**:

```python
@app.command("ip")
def show_ip():
    """Show local IP address."""
    import socket
    ip = socket.gethostbyname(socket.gethostname())
    console.print(f"[green]Local IP:[/green] {ip}")
```

### Pattern: interactive Selection (FZF)

**Input**: Pick a branch.
**Correct Output**:

```python
from jpscripts.core.ui import fzf_select
branches = [b.name for b in repo.branches]
selection = fzf_select(branches, prompt="branch> ")
if not selection:
    raise typer.Exit()
```

### Pattern: Configuration Access

**Input**: Read the notes directory.
**Correct Output**:

```python
def command(ctx: typer.Context):
    state = ctx.obj
    notes_dir = state.config.notes_dir.expanduser()
```

## 4. Testing Guidelines

- **Smoke Tests**: Every new command MUST have a corresponding entry in `tests/test_smoke.py`.
- **Mocking**: Use `monkeypatch` for system calls. Do not actually kill processes or delete files during tests.
- **Fixtures**: Use `isolate_config` from `conftest.py` to prevent overwriting the user's real `~/.jpconfig`.

## 5. Critical Constraints

- **Type Hints**: All files must utilize `from __future__ import annotations`.
- **Path Safety**: Use `pathlib.Path` exclusively. Never string manipulation for paths.
- **Imports**: lazy import heavy modules (like `git` or `pandas`) inside the function if they are not used globally, to keep CLI startup time \< 50ms.
