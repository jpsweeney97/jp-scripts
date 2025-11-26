# JPScripts Constitution (God-Mode)

Read this document before touching code. Violations halt work until corrected.

## 1) Prime Directives
- **Context is King**: Always read this constitution first, then re-check relevant files before edits.
- **Atomic Commits**: Separate refactors from feature work; never mix orthogonal changes in one commit.

## 2) Architecture
- Layering is strict: `Commands` ➜ `Core` ➜ `MCP`. Dependencies flow downward only.
- Entry point `src/jpscripts/main.py` performs registration/bootstrap only—no business logic.
- All I/O is asynchronous via `asyncio`; blocking calls must be wrapped in threads or replaced with async equivalents.
- Core libraries: Typer/Rich for CLI, Pydantic for validation, SQLite for storage. No ad-hoc dependencies.

## 3) MCP Protocol
- Tools live in `src/jpscripts/mcp/tools/` and are registered via `src/jpscripts/mcp/server.py` through `register_tools`.
- Every new MCP tool must use the `@tool` decorator and wrap execution with `@tool_error_handler`.
- File and path operations must call `security.validate_path(...)` to enforce sandbox boundaries.
- Favor structured outputs; avoid free-form text from MCP tools unless explicitly required.
- All MCP tools must rely on Pydantic runtime validation. Primitives must be fully type-hinted. The server will fail to start if untyped arguments are detected.

## 4) Testing Standards
- Async code requires `pytest-asyncio`; write tests that exercise awaitables directly.
- Every new CLI command must have a `test_smoke.py` entry to ensure `--help` and basic invocation succeed.
- Keep unit tests in `tests/unit/` and integration/system tests in `tests/integration/`; do not bypass them.
