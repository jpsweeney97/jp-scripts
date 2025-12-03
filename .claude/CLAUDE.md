# CLAUDE.md — jpscripts

> **Last updated:** 2025-12-03
> A Python CLI toolkit and AI agent framework for developer productivity.

---

## Quick Reference

```bash
# Development
make test              # Full test suite + linting
pytest                 # Fast tests only
pytest tests/unit      # Unit tests only
pytest --cov           # With coverage
mypy src               # Type checking
ruff check src         # Lint
ruff format src        # Format

# CLI (after `pip install -e ".[dev,ai]"`)
jp --help              # All commands
jp fix "..."           # AI-powered code repair
jp map                 # Generate AST repo map
jp memory search "..." # Semantic memory search
```

---

## Project Layout

```
src/jpscripts/
├── main.py              # CLI entrypoint, command discovery
├── commands/            # Typer CLI modules (nav, agent, search, etc.)
├── core/                # Shared logic (config, security, rate limiting)
├── agent/               # AI agent engine, parsing, tools
├── git/                 # Async git operations (AsyncRepo)
├── mcp/                 # MCP server and tool registry
├── providers/           # LLM backends (anthropic, openai)
├── memory/              # Semantic memory with embeddings
└── governance/          # AST checker, secret scanner, compliance

tests/
├── unit/                # Fast, isolated tests
├── integration/         # Tests hitting real services
├── security/            # Symlink attacks, injection tests
└── conftest.py          # Shared fixtures
```

---

## Conventions (Project-Specific)

### Error Handling

Use `Result[T, E]` for recoverable failures; exceptions for bugs:

```python
# ✅ Correct
from jpscripts.core.result import Ok, Err, Result

async def read_file(path: Path) -> Result[str, str]:
    if not path.exists():
        return Err(f"File not found: {path}")
    return Ok(path.read_text())

# ❌ Wrong
def read_file(path: Path) -> str:
    try:
        return path.read_text()
    except Exception:
        pass  # Never do this
```

Never use bare `except Exception: pass`. Always name exception variables `exc`.

### Layer Boundaries (Enforced)

```
commands/  →  core/, git/, providers/     ✅ allowed
commands/  →  mcp/                        ❌ forbidden (move shared logic to core/)
mcp/tools/ →  core/, git/                 ✅ allowed
providers/ →  AppConfig (direct import)  ❌ forbidden (pass config as param)
```

### Security Invariants

1. **Path validation**: All file paths must pass through `security.validate_path_safe()` before I/O
2. **Workspace sandbox**: MCP tools reject paths outside workspace root
3. **Secret redaction**: Strip API keys from errors before logging
4. **Symlink safety**: Max 10 hops, no circular chains, forbid system roots (`/etc`, `/usr`, etc.)

```python
from jpscripts.core.security import validate_path_safe

# Before any file operation
result = await validate_path_safe(user_path, workspace_root)
match result:
    case Ok(safe_path):
        # Proceed with safe_path
    case Err(msg):
        # Reject with clear error
```

### Async Purity

All I/O operations must be async. Never use blocking calls in `async def`:

```python
# ✅ Correct
async def fetch_data():
    async with aiohttp.ClientSession() as session:
        return await session.get(url)

# ❌ Wrong (blocks event loop)
async def fetch_data():
    return requests.get(url)  # Blocking!
```

### Type Strictness

- All public APIs must have complete type annotations
- Run `mypy src` before committing
- Use `TypedDict` for structured dicts, not `dict[str, Any]`

---

## Common Patterns

### Adding a New CLI Command

1. Create `src/jpscripts/commands/mycommand.py`:

```python
import typer
from jpscripts.core.config import AppConfig, get_config

app = typer.Typer()

@app.command()
def run(ctx: typer.Context) -> None:
    """Short description for --help."""
    config: AppConfig = get_config(ctx)
    # Implementation
```

2. Commands are auto-discovered from `commands/` — no registration needed.

### Adding an MCP Tool

1. Create `src/jpscripts/mcp/tools/mytool.py`:

```python
from mcp.server import Server

def register(server: Server) -> None:
    @server.call_tool()
    async def my_tool(arguments: dict) -> list:
        # Validate paths!
        # Return [TextContent(...)]
```

2. Tools are auto-discovered from `mcp/tools/` via `pkgutil`.

### Running Agent with Context

```python
from jpscripts.agent.engine import AgentEngine
from jpscripts.agent.context import ContextGatherer

context = await ContextGatherer(workspace).gather()
engine = AgentEngine(provider, config)
result = await engine.run(task, context=context)
```

---

## Testing Guidelines

- **Unit tests**: Pure logic, no I/O, mock external deps
- **Integration tests**: Real services, marked `@pytest.mark.integration`
- **Security tests**: Attack simulations in `tests/security/`

```bash
# Run security tests specifically
pytest tests/security -v

# Skip slow integration tests
pytest -m "not integration"
```

Test file naming: `test_{module}.py` mirrors source structure.

---

## External Docs (Don't Duplicate Here)

| Topic                                   | Location                                         |
| --------------------------------------- | ------------------------------------------------ |
| Development setup, PR workflow          | `CONTRIBUTING.md`                                |
| Architecture diagrams, design rationale | `docs/ARCHITECTURE.md`                           |
| Adding commands/tools/providers         | `docs/EXTENDING.md`                              |
| CLI usage, options, examples            | `docs/CLI_REFERENCE.md`                          |
| Completed refactor roadmap              | `docs/UNIFIED_ARCHITECTURAL_REFACTOR_ROADMAP_COMPLETED.md` |

---

## Audit Mode

When asked to "run a tech debt audit" or "health check this repo," produce structured output:

1. **Overview**: What you inspected, assumptions made
2. **Findings**: By category (Architecture, Code Quality, Testing, Security, etc.)
3. **Prioritized Roadmap**: P0 (critical) → P1 (important) → P2 (cleanup)
4. **Deep Dives**: Top 3-5 issues with concrete refactor plans
5. **Task Prompts**: Copy-pastable Claude Code tasks

Reference the 155-objective health checklist in prior audit artifacts when relevant:

- `TECH_DEBT_AUDIT_2025-12-01_COMPLETED.md`
- `TECH_DEBT_AUDIT_2025-11-30_COMPLETED.md`

---

## Session Continuity

If working on a multi-phase roadmap:

1. Read the active roadmap at session start (currently: none active)
2. Check the Progress Tracker for current phase/step
3. Resume from recorded position — don't restart completed work

Completed roadmaps:
- `docs/UNIFIED_ARCHITECTURAL_REFACTOR_ROADMAP_COMPLETED.md` (8 phases, 2025-12-03)
- `docs/ARCHITECTURE_HARDENING_ROADMAP_COMPLETED.md` (4 phases, 2025-12-02)
- `docs/CODE_QUALITY_ROADMAP_COMPLETED.md` (6 phases, 2025-12-02)

---

## Guardrails

- Never bypass permission prompts or sandboxing
- No pushes to remote branches without explicit instruction
- No rewrites of git history or CI/CD configs
- Treat `.env` and `secrets/**` as highly sensitive
- If a file is unseen, say "Not Found" — don't hallucinate

## Prompt Engineering

@.claude/PROMPT_TEMPLATE.md
