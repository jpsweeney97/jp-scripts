# jpscripts Constitution & Agent System Prompt

> **Version:** 2.0
> **Owner:** jp-scripts maintainers
> **Purpose:** Codify non-negotiable engineering standards for agents and humans.
> **Scope:** All code, tests, and tools within the jpscripts workspace.

---

## Core Invariants

You MUST adhere to these invariants at all times. Any violation requires explicit approval.

### [invariant:typing] Strict Typing

- All code must pass `mypy --strict`
- Every function, method, and attribute carries explicit type annotations, including `-> None` for procedures
- **Any use of `Any` is prohibited** unless wrapping third-party APIs without stubs
  - When unavoidable, justify with a `# type: ignore` comment plus runtime validation at the boundary
- All MCP tool parameters and configuration objects use `pydantic.BaseModel` or `dataclass`
- No raw dicts for structured data

### [invariant:async-io] Async-First I/O

- All I/O (git, filesystem, subprocess, HTTP) executes in `async def`
- Blocking calls must be wrapped in `asyncio.to_thread`
- Shell execution **never** uses `shell=True`
- Commands are tokenized with `shlex.split` and executed via:
  - `asyncio.create_subprocess_exec`
  - `core.system.run_safe_shell`
- All shell-capable code must invoke `core.command_validation.validate_command` before execution

### [invariant:error-handling] Result-Based Errors

- **No bare `except` clauses**
- Catch specific exceptions and map to `Result[T, JPScriptsError]` variants
- Core layers return `Result` without printing
- Command layers pattern-match `Ok`/`Err` and render via console
- MCP layers use `tool_error_handler`
- Raw stack traces are never shown to users; errors are wrapped in typed domain errors

### [invariant:context-and-memory] Smart Context Management

- Structured content (`.py`/`.json`/`.yaml`) is never naively sliced
- Use `smart_read_context` and `TokenBudgetManager` for truncation
- For Python code, use `DependencyWalker` for semantic slicing
- File memories include content hashes
- Prune memory entries on drift or deletion
- Retrieval uses reciprocal rank fusion (k=60)

### [invariant:testing] Comprehensive Testing

- Every CLI command has a smoke test in `tests/test_smoke.py` covering `--help` and basic invocation
- Async tests use `pytest.mark.asyncio`
- Security-sensitive code ships explicit security tests
- Bug fixes include regression coverage
- Follow the Reflexion Protocol: **Plan → Test (Red) → Execute (Green) → Verify**

### [invariant:destructive-fs] Protected Filesystem Operations

Destructive Python file operations are forbidden unless explicitly marked safe:

```python
# Forbidden without marker:
shutil.rmtree(path)
os.remove(path)

# Allowed with safety marker:
shutil.rmtree(path)  # safety: checked
```

This is enforced by AST analysis via `SecurityVisitor`.

### [invariant:dynamic-execution] No Dynamic Execution

The following are **strictly forbidden** to prevent obfuscated shell injection:
- `eval()`
- `exec()`
- Dynamic imports via string manipulation

---

## Cognitive Standards

Apply these mental checks before every action.

### Safety Scan

Before executing any tool, performing any write, or running any command:
1. **List potential side effects** (file modification, process termination, network calls)
2. **Verify reversibility** - can this be undone?
3. **Check authorization** - is this within allowed operations?

### Invariant Citation

Every step in a proposed plan must explicitly cite the Constitution Invariant ID it satisfies:

```
Step 1: Create async subprocess wrapper
        [invariant:async-io] - Uses asyncio.create_subprocess_exec
        [invariant:typing] - Returns Result[str, ProcessError]
```

### Anti-Pattern Check

Before finalizing any response, verify against known anti-patterns:

| Anti-Pattern | Detection | Remediation |
|-------------|-----------|-------------|
| Sync I/O | `subprocess.run()` without `asyncio.to_thread` | Wrap in async context |
| Shell Injection | `shell=True`, `os.system()` | Use `shlex.split` + `subprocess_exec` |
| Bare Except | `except:` without type | Specify exception types |
| Naive Slicing | `content[:limit]` on code | Use `smart_read_context` |
| Any Type | `def foo() -> Any` | Add proper type annotations |

---

## Protocols

### Reflexion Protocol

```
Plan → Test (Red) → Execute (Green) → Verify
```

**On first failure:**
- Retry with a minimal adjustment

**On second failure:**
1. HALT execution
2. Perform structured reflection:
   - Identify root causes
   - Question assumptions
   - Choose an alternate approach
3. Document the reflection
4. Proceed with revised plan

### Commits and Edits Protocol

- **Never amend or revert** user changes unless explicitly requested
- Validate file existence before reading or writing
- Prefer idempotent operations
- Avoid destructive commands (`reset`, `checkout`, `rm`) without explicit approval
- When creating commits:
  - Use conventional commit messages
  - Include `[invariant:*]` markers for modified code

### Command Execution Protocol

1. **Validate:** `core.command_validation.validate_command(cmd, cwd)`
2. **Tokenize:** `shlex.split(cmd)` - never construct from raw strings
3. **Execute:** `asyncio.create_subprocess_exec(*tokens, ...)`
4. **Warn:** When commands reference paths outside workspace

### Protocol: Self-Repair

- **Dependency Conflicts**: If import errors occur, run `uv pip install -e .` or `pip install -e .` immediately.
- **Type Narrowing**: If `mypy` flags a union type, assert the type explicitly before usage. Do not cast `Any`.
- **Context Exhaustion**: If context limit is reached, call `tools.read_file_paged` on only the critical sections of files.

---

## Security Clearance

### Allowed Operations

✅ Read operations on paths validated by `security.validate_path` within workspace root
✅ Write operations limited to workspace and configured writable roots
✅ External writes require explicit approval
✅ Async subprocess execution after command validation
✅ Use `shlex.split` for command tokenization

### Forbidden Operations

❌ Following symlinks outside the workspace
❌ Naive truncation of structured files
❌ Exposing raw stack traces to users
❌ Executing `shell=True` or `os.system()`
❌ Unsafe path traversal (e.g., `../../../etc/passwd`)
❌ Reading sensitive files (`.env`, `*.pem`, `*.key`, `id_rsa`)

---

## Parallel Swarm Execution

When operating in parallel swarm mode:

### Worktree Isolation

Each parallel task runs in its own git worktree to prevent:
- Git `index.lock` contention
- Filesystem race conditions
- Merge conflicts during parallel execution

### DAG-Based Orchestration

Tasks are organized as a Directed Acyclic Graph (DAG):
- Tasks with disjoint file sets can run in parallel
- Tasks with shared files must be sequenced
- Use `DAGGraph.detect_disjoint_subgraphs()` for parallel grouping

### Merge Strategy (Moderate)

| Conflict Type | Strategy |
|--------------|----------|
| TRIVIAL (whitespace, imports) | Auto-resolve |
| SEMANTIC (logic changes) | Attempt LLM resolution |
| COMPLEX (structural overlap) | Flag for human review |

---

## Quick Reference

```
# Check a command
from jpscripts.core.command_validation import validate_command
verdict, reason = validate_command(cmd, cwd)

# Smart context reading
from jpscripts.core.context_gatherer import smart_read_context
content = smart_read_context(path, max_chars, max_tokens)

# Semantic code slicing
from jpscripts.core.dependency_walker import DependencyWalker
walker = DependencyWalker(source)
slice = walker.slice_to_budget("main", max_tokens=1000)

# Async subprocess
async def run_cmd(cmd: str, cwd: Path) -> str:
    tokens = shlex.split(cmd)
    proc = await asyncio.create_subprocess_exec(
        *tokens, cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return stdout.decode()

# Result pattern
match await async_operation():
    case Ok(value):
        process(value)
    case Err(error):
        handle_error(error)
```

---

*This constitution is automatically enforced by `SecurityVisitor` in `governance.py`. Violations will be flagged during code review and CI/CD.*
