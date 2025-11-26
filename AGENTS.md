# JPScripts Constitution & Architectural Guidelines

You are contributing to `jpscripts`, a mission-critical Python CLI OS.
Follow these invariants strictly.

## 1. Architecture & Boundaries

- **Unidirectional Flow**: `Commands` import `Core`. `Core` never imports `Commands`.
- **Entry Point**: `src/jpscripts/main.py` is for registration only. No logic allowed.
- **Core Libraries**:
  - **CLI**: `typer` (commands), `rich` (output).
  - **Data**: `pydantic` (validation), `sqlite3` (storage).
  - **Async**: `asyncio` is mandatory for all I/O.

## 2. Coding Standards

- **Type Safety**: `mypy --strict` compliance is required. Use `from __future__ import annotations`.
- **Error Handling**:
  - Never use bare `except:`.
  - User-facing errors must be caught and printed nicely via `console.print(..., style="red")`.
  - Raw stack traces are only for `--verbose` mode.
- **Path Safety**: All file operations must check `security.validate_path(path, root)` to prevent directory traversal.

## Performance Invariants

- File system traversal must utilize `ripgrep` or `fd` via subprocess. Python-native walking is banned for depth > 2.

## 3. Agent Interaction Strategy

- **Context**: When modifying a file, first check its imports to understand dependencies.
- **tools**: Prefer using the specific MCP tools (`search_codebase`, `read_file`) over raw shell commands (`grep`, `cat`).
- **Testing**:
  - Unit tests go in `tests/unit/`.
  - Integration tests go in `tests/integration/`.
  - **Mandatory**: Every new feature must have a `test_smoke.py` entry ensuring the CLI command runs without crashing.
- **Validation**:
  - After each file modification, run tests.

## 4. File System & Git

- **Performance**: Do not list all files in the repo. Use `git ls-files` or `rg` for discovery.
- **Safety**: Never run `git push` or `git clean -fdx` without explicit user confirmation flags (`--force` / `-y`).

## 5. Documentation

- Update `CHANGELOG.md` with every feature addition.
- Docstrings must be Google-style.

## Agent Protocol

- All Agent interactions must use Structured JSON Outputs enforced by Pydantic schemas. No Regex parsing of natural language responses.

## 6. The Self-Correction Protocol

When acting in an autonomous loop (`jp fix --loop`):

1. **Verify First**: Never declare a task complete without running the verification command provided.
2. **Incrementalism**: If a fix fails, do not hallucinate a new library. Revert to first principles. Read the error message literally.
3. **Output Format**: When requested, provide _only_ the raw Git Patch or diff code block to minimize parsing errors.
