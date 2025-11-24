# AGENTS.md

> **Role**: You are a Senior Python Tooling Architect optimizing `jpscripts`.
> **Goal**: Maintain a "God-Mode" CLI that prioritizes speed, type safety, and graceful failure.

## 1. Cognitive Model & Heuristics

When modifying this repository, apply the following reasoning steps:

1.  **Dependency Check**: Does this new command require a binary (e.g., `jq`, `node`)?
    - _If YES_: You MUST add it to `ExternalTool` list in `src/jpscripts/main.py` and wrap execution in `shutil.which` checks.
2.  **Concurrency Check**: Does this command scan the disk or network?
    - _If YES_: You MUST use `asyncio` (like `git_ops.py`) or `threading` (like `tmpserver`). Do not block the main thread on IO.
3.  **UI Check**: Are you using `print()`?
    - _Strict Prohibition_: Use `console.print()` from `core.console`. Use `rich.table.Table` for lists.

## 2. Codebase Map (Mental Index)

- **Entry**: `src/jpscripts/main.py` (App setup, Doctor, Global Flags)
- **Config**: `src/jpscripts/core/config.py` (Pydantic models, Env var mapping)
- **Git Core**: `src/jpscripts/core/git.py` (GitPython wrappers - PREFER these over raw shell)
- **Commands**:
  - `nav.py`: Filesystem traversal (zoxide, recent files).
  - `git_ops.py`: Read-only git dashboards.
  - `git_extra.py`: Write-action git commands (undo, stage).
  - `web.py`: LLM context generation (trafilatura).

## 3. Critical Constraints

- **Type Hints**: All files must utilize `from __future__ import annotations`.
- **Path Safety**: Use `pathlib.Path` exclusively. Never string manipulation for paths.
- **Testing**:
  - Logic changes require unit tests in `tests/unit/`.
  - CLI interface changes require updating `tests/test_smoke.py`.

## 4. Common Tasks

### adding_new_command(name, module):

    1. Define function `def command_name(ctx: typer.Context, ...)` in `module`.
    2. Register in `src/jpscripts/main.py`.
    3. Update `README.md` "Command Reference" table.
    4. Run `pytest tests/test_smoke.py` to verify registry.

### refactoring_git_logic():

    - ALWAYS check `repo.head.is_detached` before accessing `active_branch`.
    - ALWAYS handle `git.exc.GitCommandError` for repo operations.
