# JPScripts Constitution (God-Mode)

**Read this document before touching code. Violations halt work until corrected.**

This constitution is the definitive source of truth for all architectural decisions, coding standards, and invariants enforced within the `jpscripts` codebase. Every contributor—human or agent—must internalize these rules.

---

## Section 1: Type Safety

### 1.1 Strict Type Enforcement

- **Mypy Strict Mode**: All code must pass `mypy --strict`. No exceptions.
- **Full Signatures**: Every function, method, and class attribute must have explicit type annotations.
- **Return Types**: Never omit return type annotations. Use `-> None` explicitly for procedures.

### 1.2 Prohibition of `Any`

- **`Any` is Forbidden** unless wrapping a third-party library that lacks type stubs.
- When `Any` is unavoidable, it MUST be accompanied by:
  1. A comment explaining why (`# type: ignore[...] - reason`)
  2. Runtime validation at the boundary

### 1.3 Pydantic Validation

- **MCP Tools**: All MCP tool parameters must use Pydantic models or fully type-hinted primitives. The server will fail to start if untyped arguments are detected.
- **Agent Exchanges**: All data flowing into/out of `AgentEngine` must be validated via Pydantic models (`AgentResponse`, `ToolCall`, etc.).
- **Configuration**: Use `pydantic.BaseModel` or `dataclass` for all configuration objects. No raw dicts for structured data.

---

## Section 2: Async Purity & I/O

### 2.1 The Golden Rule

> **All I/O operations (Git, HTTP, Filesystem) MUST be `async def`.**

There are no exceptions. Synchronous I/O in an async context is a correctness bug.

### 2.2 Blocking Call Prohibition

- **Forbidden Pattern**: Calling blocking functions (e.g., `requests.get`, `subprocess.run`, `open().read()`) inside `async def`.
- **Required Pattern**: Wrap unavoidable blocking calls in `asyncio.to_thread(...)`.

```python
# WRONG
async def fetch_data():
    return requests.get(url)  # BLOCKS THE EVENT LOOP

# CORRECT
async def fetch_data():
    return await asyncio.to_thread(requests.get, url)
```

### 2.3 Shell Execution Security

**Direct shell execution (`shell=True`) is an IMMEDIATE FAILURE CONDITION.**

| Forbidden                         | Required                                    |
| --------------------------------- | ------------------------------------------- |
| `subprocess.run(cmd, shell=True)` | `asyncio.create_subprocess_exec(*tokens)`   |
| `os.system(cmd)`                  | `jpscripts.core.system.run_safe_shell(cmd)` |
| Raw string commands               | `shlex.split(cmd)` for tokenization         |

### 2.4 Command Validation

- All functions capable of executing shell commands MUST invoke `core.command_validation.validate_command()` before execution.
- Context gatherers, tool executors, and any shell-adjacent code must validate commands.
- Silent execution without validation is a **security vulnerability**.

---

## Section 3: Error Containment

### 3.1 No Bare Exceptions

```python
# FORBIDDEN
except:
    pass

except Exception:
    print("error")

# REQUIRED
except SpecificError as exc:
    return Err(ConfigurationError("context", context={"error": str(exc)}))
```

### 3.2 Result-Based Error Handling

- **Core Logic**: Must return `Result[T, JPScriptsError]` for all I/O and external calls.
- **Pattern Matching**: Command and MCP layers must pattern-match on `Ok(...)` / `Err(...)`:

```python
match some_operation():
    case Ok(value):
        process(value)
    case Err(error):
        handle_error(error)
```

### 3.3 Layer Separation

| Layer                      | Error Handling                                                  |
| -------------------------- | --------------------------------------------------------------- |
| **Core** (`core/`)         | Returns `Result[T, Error]`. NEVER prints to stdout/stderr.      |
| **Commands** (`commands/`) | Pattern-matches on Results. Renders errors via `console.print`. |
| **MCP** (`mcp/`)           | Uses `@tool_error_handler`. Returns structured error responses. |

### 3.4 No Raw Stack Traces

- CLI must never expose raw Python tracebacks to users.
- All exceptions must be caught and wrapped in `jpscripts.core.result.Result` types or handled via `@handle_exceptions` decorator.

---

## Section 4: Context & Memory

### 4.1 No Naive Truncation

**Naive string slicing of `.py`, `.json`, or any structured file is STRICTLY FORBIDDEN.**

- **Required**: Use `smart_read_context` (AST-aware) when truncating source code.
- **Context Budgeting**: Use `TokenBudgetManager` to allocate tokens by priority:
  1. Diagnostics (errors, warnings)
  2. Diffs (git changes)
  3. Files (source code)

### 4.2 Memory Hygiene

- **Content Hashing**: File-based memories MUST include `content_hash` (MD5) of the source file at save time.
- **Drift Detection**: `prune_memory` removes entries where:
  1. Source file no longer exists
  2. Source file hash differs from stored hash (content drift)
- **Backward Compatibility**: Entries without hashes (pre-upgrade) are treated as valid.

### 4.3 Memory Retrieval

- Retrieval uses **Reciprocal Rank Fusion (RRF)** with `k=60`.
- Do not modify scoring constants without empirical justification and benchmarks.

---

## Section 5: Testing Strategy

### 5.1 Smoke Tests

- **Every CLI command** must have a smoke test entry in `tests/test_smoke.py`.
- Smoke tests verify:
  1. `--help` outputs without error
  2. Basic invocation succeeds (or fails gracefully with expected error)

### 5.2 Async Testing

- All async code must be tested with `pytest-asyncio`.
- Use `@pytest.mark.asyncio` decorator for async test functions.
- Test awaitables directly—do not wrap in `asyncio.run()` inside tests.

### 5.3 Test Organization

| Directory            | Purpose                                                     |
| -------------------- | ----------------------------------------------------------- |
| `tests/unit/`        | Unit tests for individual functions/classes                 |
| `tests/integration/` | Integration tests involving multiple components             |
| `tests/security/`    | Security-focused tests (command validation, path traversal) |

### 5.4 Coverage Requirements

- New features must include tests.
- Bug fixes should include regression tests.
- Security-sensitive code requires explicit security tests.

---

## Section 6: Cognitive Standards

### 6.1 Reflexion Required

Agents must not simply loop on errors. The reflexion protocol:

1. **First Failure**: Retry with minor adjustment.
2. **Second Failure**: HALT. Force "Step Back" strategy:
   - Analyze root cause
   - Question assumptions
   - Consider alternative approaches
3. **After Reflection**: Resume with "Step Forward" (execution).

### 6.2 Structured Thought

**Stream-of-consciousness logs are not accepted.** All agent thoughts must follow:

```
Observation → Critique → Plan → Action
```

| Phase           | Purpose                                            |
| --------------- | -------------------------------------------------- |
| **Observation** | What do I see? What is the current state?          |
| **Critique**    | What could go wrong? What assumptions am I making? |
| **Plan**        | What steps will I take? In what order?             |
| **Action**      | Execute the plan. One step at a time.              |

### 6.3 Tool Hygiene

- **Verify Before Access**: Agents must verify file existence via `ls` or `list_directory` before `read_file` or `apply_patch`.
- **Blind Access Forbidden**: Attempting to read/modify a file without verification is a protocol violation.
- **Idempotency**: Prefer idempotent operations. Check state before modifying.

---

## Appendix A: Architecture Overview

### Layering

```
Commands → Core → MCP
    ↓        ↓      ↓
   CLI    Logic   Tools
```

Dependencies flow **downward only**. Core never imports from Commands.

### Entry Point

- `src/jpscripts/main.py` performs registration/bootstrap only—no business logic.
- Commands are dynamically discovered via `pkgutil` from `src/jpscripts/commands/`.

### The Unified Engine

- All agentic workflows (single or swarm) MUST use `jpscripts.core.engine.AgentEngine`.
- No ad-hoc prompt loops or subprocess calls in command modules.
- `AgentEngine` persists execution traces to `~/.jpscripts/traces`.

### Provider Architecture

- **No Subprocess Inception**: Direct shelling out to external CLIs is forbidden in core logic.
- **Exception**: `jpscripts.providers.codex` is the designated Legacy Adapter for Codex CLI.
- All other modules must use the `LLMProvider` interface.

---

## Appendix B: Capability Tiers

### Core Dependencies (Zero Optional Deps)

- Git operations
- Navigation utilities
- Filesystem operations
- Process management

Failures surface as typed errors: `Err(GitError | NavigationError | SystemResourceError)`.

### AI Dependencies (Optional)

- Embeddings (sentence-transformers)
- Vector store (LanceDB)
- Agent scaffolding

Missing capabilities raise `CapabilityMissingError` instead of silently downgrading.

---

## Appendix C: MCP Protocol

### Tool Registration

- Tools live in `src/jpscripts/mcp/tools/`.
- Registration via `src/jpscripts/mcp/server.py` through `register_tools`.
- Use `@tool` decorator and `@tool_error_handler` wrapper.

### Security

- File/path operations must call `security.validate_path(...)` to enforce sandbox boundaries.
- Tools exposed to MCP must be identical to tools registered in `AgentEngine`.

### Output Standards

- Favor structured outputs (JSON, typed models).
- Avoid free-form text unless explicitly required.

---

## Enforcement

These rules are enforced through:

1. **CI Pipeline**: `mypy --strict`, `ruff check`, `pytest`
2. **Pre-commit Hooks**: Type checking, linting
3. **Code Review**: Human and agent reviewers verify compliance
4. **Runtime Validation**: Pydantic models, command validation

**Violations halt work until corrected. No exceptions.**
