# AGENTS.md

> **Role**: You are a Principal Software Engineer maintaining `jpscripts`.
> **Context**: This is a Python 3.11+ Typer CLI managed with `hatch`.
> **Philosophy**: Speed (<50ms startup), Type Safety (Strict MyPy), and Modularity.

## 1. Map of the Territory

To save context, assume this architecture without reading files:

- **Entry**: `src/jpscripts/main.py` initializes `AppState` and Typer.
- **Config**: `src/jpscripts/core/config.py` loads `~/.jpconfig` (TOML) + Env Vars.
- **Git Core**: `src/jpscripts/core/git.py` wraps `GitPython`. **Invariant**: Never use `subprocess` for git ops unless `GitPython` is insufficient.
- **Commands**:
  - `nav.py`: File system traversal (Zoxide, recents).
  - `git_ops.py`: Async status checks.
  - `context.py`: LLM context packing.
  - `system.py`: OS interactions (processes, ports).

## 2. Review Guidelines (Codex Strict Mode)

Codex, when running `@codex review`, prioritize these findings:

- **P0 (Blocking)**:

  - **XML/Injection Safety**: Ensure content packed for LLMs or Shells is escaped.
  - **Blocking I/O**: Any synchronous file/network IO inside `async def` functions.
  - **Type Safety**: Any usage of `Any` or missing return type annotations.
  - **Path Safety**: Use `pathlib.Path` exclusively. No string concatenation for paths.

- **P1 (High Priority)**:
  - **Missing Docs**: New commands must have a docstring.
  - **Fragile Imports**: Top-level imports of heavy libraries (`git`, `pandas`) must be deferred to the function scope to protect startup time.

## 3. Cognitive Model & Decision Tree

1.  **Dependency Check**:

    - Need a binary (e.g., `gh`, `fzf`)? -> Check `DEFAULT_TOOLS` in `main.py`.
    - Need a library? -> Check `pyproject.toml`. Do not add heavy deps without approval.

2.  **Concurrency Check**:

    - Operation > 100ms? -> Must use `asyncio` and `rich.live.Live` or `rich.progress`.
    - File system scan? -> Use iterators, do not load full lists into memory.

3.  **UI Check**:
    - **PROHIBITED**: `print()`, `input()`.
    - **REQUIRED**: `console.print()`, `rich.prompt.Prompt`.

## 4. Architectural Invariants & Cognitive Model

### The "Command Pattern" Heuristic

When creating a new command, the Agent MUST follow this mental graph:

1.  **Input**: Define strict `typer.Option` types. Never use `input()`.
2.  **Logic**: If logical complexity > 5 lines, move to `src/jpscripts/core/{domain}.py`.
3.  **Output**: Use `console.print` with Rich markup.
4.  **Registration**: The Agent MUST automatically register the command in `src/jpscripts/main.py` or the build is broken.

### Context Awareness

- **Configuration**: The Agent must never hardcode paths. Always resolve via `ctx.obj.config`.
- **Git Operations**: Use `jpscripts.core.git`. Do not shell out to `subprocess.run("git", ...)` unless `GitPython` lacks the specific plumbing (e.g., complex interactive rebase).
