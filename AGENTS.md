# AGENTS.md

> **Role**: You are a Principal Software Engineer maintaining `jpscripts`.
> **Context**: This is a Python 3.11+ Typer CLI.
> **Philosophy**: Speed (<50ms startup), Type Safety (Strict MyPy), and Modularity.

## Review guidelines

Codex, when running `@codex review`, you must prioritize findings as follows:

- **P0 (Blocking)**:

  - **Blocking I/O**: Any synchronous file/network IO inside `async def` functions.
  - **Shell Injection**: Usage of `subprocess.run(shell=True)` or unsanitized input in `shlex`.
  - **Type Safety**: Any usage of `Any` or missing return type annotations.
  - **Path Safety**: String concatenation for paths instead of `pathlib.Path / "child"`.

- **P1 (High Priority)**:
  - **Missing Docs**: New commands without a docstring or `README.md` entry.
  - **Missing Smoke Test**: New commands not registered in `tests/test_smoke.py`.
  - **Fragile Imports**: Top-level imports of heavy libraries (`git`, `pandas`) that slow down CLI startup.

## 1. Cognitive Model & Heuristics

When writing code for this repository, you must adhere to this decision tree:

1.  **Dependency Check**:

    - Does the command need a binary (e.g., `gh`, `fzf`, `rg`)?
    - **ACTION**: Add it to `DEFAULT_TOOLS` in `src/jpscripts/main.py`. Wrap execution in `shutil.which`.

2.  **Concurrency Check**:

    - Does the command touch the disk more than 10 times or hit the network?
    - **ACTION**: Use `asyncio`. Use `rich.live.Live` for status updates (see `git_ops.py`).

3.  **UI Check**:
    - **PROHIBITED**: `print()`, `input()`.
    - **REQUIRED**: `jpscripts.core.console.console.print()`, `rich.prompt.Prompt`.

## 2. Project Architecture

| Path                      | Purpose                                                                |
| :------------------------ | :--------------------------------------------------------------------- |
| `src/jpscripts/main.py`   | App entry, config loading, dependency injection (`AppState`).          |
| `src/jpscripts/core/`     | **Stable Kernel**. Changes here require P0 review.                     |
| `src/jpscripts/commands/` | **Feature Plugins**. Isolated logic. Must use `ctx.obj` for config.    |
| `tests/test_smoke.py`     | **CI Gate**. Checks that every command runs `--help` without crashing. |

## 3. Implementation Patterns

### Pattern: Fast-Path Imports

**Goal**: Keep `jp --help` under 50ms.
**Wrong**:

```python
import pandas as pd  # Slows down EVERY command
def analyze(): ...
```

**Right**:

```python
def analyze():
    import pandas as pd  # Import only when needed
    ...
```

### Pattern: Configuration Injection

**Goal**: Never read env vars directly in commands.
**Right**:

```python
def my_cmd(ctx: typer.Context):
    state: AppState = ctx.obj
    # Config is already loaded, validated, and normalized
    target = state.config.workspace_root
```
