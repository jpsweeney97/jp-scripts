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
- Commands are dynamically discovered. To add a new command, create a file in `src/jpscripts/commands/` and define a module-level `app = typer.Typer()` object.
- **No Subprocess Inception**: Never shell out to the `jp` or `codex` CLI from within Python code. Use internal modules (`core`, `providers`) directly.

## 3) MCP Protocol

- Tools live in `src/jpscripts/mcp/tools/` and are registered via `src/jpscripts/mcp/server.py` through `register_tools`.
- Every new MCP tool must use the `@tool` decorator and wrap execution with `@tool_error_handler`.
- File and path operations must call `security.validate_path(...)` to enforce sandbox boundaries.
- Favor structured outputs; avoid free-form text from MCP tools unless explicitly required.
- All MCP tools must rely on Pydantic runtime validation. Primitives must be fully type-hinted. The server will fail to start if untyped arguments are detected.
- Tools exposed to MCP must be identical to the tools registered in `AgentEngine`. The engine is the single source of truth for tool definitions.

## 4) Testing Standards

- Async code requires `pytest-asyncio`; write tests that exercise awaitables directly.
- Every new CLI command must have a `test_smoke.py` entry to ensure `--help` and basic invocation succeed.
- Keep unit tests in `tests/unit/` and integration/system tests in `tests/integration/`; do not bypass them.

## 5) The Unified Engine

- All agentic workflows (single or swarm) MUST use `jpscripts.core.engine.AgentEngine`. Do not write ad-hoc prompt loops or subprocess calls in command modules.

## Memory

- Memory retrieval uses Reciprocal Rank Fusion (RRF). Do not modify the scoring constants `k=60` without empirical justification.

## Security & Shell Execution

- Direct shell execution (`shell=True`) is STRICTLY FORBIDDEN.
- All subprocess calls must use `asyncio.create_subprocess_exec` with tokenized argument lists.
- Use `shlex.split()` to parse command strings before execution.

## Observability

- All Agent interactions must be traceable.
- `AgentEngine` MUST persist execution traces to `~/.jpscripts/traces`.
- All agentic workflows must be inspectable via `jp trace`.

## Swarm Architecture

- Swarms run on Dynamic Personas, not fixed roles. Personas define name, style, and color; avoid hardcoding role enums.

## 6) Dynamic Architecture Invariants

- **Context Budgeting**: Never blindly truncate context. Use `TokenBudgetManager` to allocate tokens by priority (Diagnostics > Diffs > Files).
- **Syntax Preservation**: When truncating source code, ALWAYS use `smart_read_context` (AST-aware) to prevent syntax corruption. Naive string slicing of `.py` or `.json` files is **STRICTLY FORBIDDEN**.
- **Dynamic Discovery**: Do not maintain hardcoded lists of modules (e.g., for Tools or Commands). Use `pkgutil` or `importlib` to discover capabilities at runtime. The filesystem is the source of truth.
