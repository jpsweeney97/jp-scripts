# JPScripts God-Mode Handbook

Concise rules for operating `jp` with precision. No fluff, no drift.

> **Version:** 2.0 | **Python:** 3.12+ | **Typing:** mypy --strict

---

## The Zero-State (≤3 minutes to operational)

1) **Prereqs**: Python 3.12+, `git`, `ruff`, `rg`, `fzf`. Optional: `uv` or `pipx` for isolated install.  
2) **Bootstrap**:  
   - `git clone https://github.com/.../jp-scripts.git` (or sync your fork)  
   - `cd jp-scripts`  
   - `uv tool install .` _or_ `pipx install .` (fallback: `pip install -e .[dev]`)  
3) **Config**: `jp init` to generate `~/.jpconfig`; set `workspace_root` and `memory_store`.  
4) **Verify**: `jp doctor` then `jp status-all`. You are live.

---

## The OODA Loop (Observe → Orient → Decide → Act)

- **Observe**: `jp map --depth 3` for structure; `jp recent` for hot files.  
- **Orient**: `jp status-all` for repo health; `jp doctor` if tools misbehave.  
- **Decide**: `jp fix --recent "State objective"` chooses the target; on failure it escalates context (see Dynamic Context Expansion).  
- **Act**: `jp watch` runs God-Mode maintenance (syntax gate + live embedding refresh); keep it running in a dedicated pane.

---

## Dynamic Context Expansion (jp fix / jp agent)

Retries get smarter, not just louder:
- **Attempt 1 – Fast**: Run command, capture diagnostics, include directly referenced files.  
- **Attempt 2 – Deep**: Parse new errors, add imported dependencies and referenced modules. System notice injected to analyze cross-module interactions.  
- **Attempt 3 – Step Back**: Tool use disabled for the turn, reasoning_effort=high, demand a Root Cause Analysis plan before patching.  
Result: slower per retry, higher precision, fewer insanity loops.

---

## God-Mode Watch (jp watch)

- Monitors `workspace_root` via `watchdog`, honoring `ignore_dirs`.
- On `.py` save: `ruff check --select E9,F821` (syntax gate); red alert on failure.
- On text save: debounced (5s) LanceDB embedding refresh so `jp memory search` is always current.
- Live dashboard (`rich.live`) displays recent events and task status. Leave it running.

---

## The Evolve Loop (jp evolve)

Autonomous technical debt reduction that identifies, optimizes, and PRs improvements.

### The Evolution Protocol
- Autonomous refactoring is prohibited unless the test suite is green.
- `jp evolve` must run targeted pytest verification after applying any patch and abort on failure.
- Branches with failing tests must not be pushed or PR'd; reset to `main` on regression.
- PR bodies must document the verification command and its exit code.

### How It Works

1. **Debt Analysis**: Computes McCabe cyclomatic complexity for all Python files.
2. **Frequency Weighting**: Queries memory for files with frequent fix history.
3. **Debt Score**: `Complexity x (1 + Fix_Frequency)` identifies highest-value targets.
4. **Optimization**: Launches "Optimizer" persona to reduce complexity.
5. **PR Workflow**: Creates branch, applies changes, pushes, creates PR for review.

### Usage

```bash
# Analyze without changes
jp evolve run --dry-run

# Run optimization (creates PR when successful)
jp evolve run --threshold 15

# Use specific model
jp evolve run --model claude-opus-4-5

# Show complexity report only
jp evolve report

# Show debt scores
jp evolve debt
```

### Constraints

- Only runs on clean git state (no uncommitted changes)
- Respects all constitutional rules (AGENTS.md)
- Preserves public interfaces (pure refactoring)
- All changes must pass `mypy --strict`

### Security Boundaries
- Dynamic execution is banned. Do not use `eval`, `exec`, `compile`, `__import__`, or dynamic `importlib.import_module` without an explicit, reviewed `# safety: checked` override.
- Shell execution stays tokenized and validated; `shell=True` and obfuscated commands are forbidden.

---

## Parallel Swarm Execution

Execute multiple agents in parallel with git worktree isolation.

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                 ParallelSwarmController              │
├─────────────────────────────────────────────────────┤
│  DAGGraph → topological sort → parallel batches      │
│                                                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐              │
│  │Worktree │  │Worktree │  │Worktree │  (max_parallel)
│  │ task-001│  │ task-002│  │ task-003│              │
│  └────┬────┘  └────┬────┘  └────┬────┘              │
│       │            │            │                   │
│       └────────────┴────────────┘                   │
│                    │                                │
│            MergeConflictResolver                    │
│       TRIVIAL → SEMANTIC → COMPLEX                  │
└─────────────────────────────────────────────────────┘
```

### DAG Task Model

```python
DAGTask(
    id="task-001",
    objective="Implement feature X",
    files_touched=["src/foo.py", "src/bar.py"],
    depends_on=["task-000"],  # Must complete first
    persona="engineer",       # or "qa"
    priority=10,              # Higher = first in batch
    estimated_complexity="moderate",
)
```

### Worktree Isolation

Each task runs in its own git worktree:
- **Prevents** `index.lock` contention
- **Prevents** filesystem race conditions
- **Enables** true parallel git operations

```python
async with manager.create_worktree("task-001") as ctx:
    # ctx.worktree_path - isolated checkout
    # ctx.branch_name   - unique branch (swarm/task-001-abc123)
    await execute_task(ctx)
# Automatic cleanup on exit
```

### Merge Strategy (3-Tier)

| Category | Detection | Resolution |
|:---------|:----------|:-----------|
| `TRIVIAL` | Whitespace-only, import reordering | Auto-resolve deterministically |
| `SEMANTIC` | Logic changes in non-overlapping regions | Attempt LLM-assisted resolution |
| `COMPLEX` | Structural overlap, high divergence | Flag for human review |

### Usage

```python
from jpscripts.swarm import ParallelSwarmController
from jpscripts.core.dag import DAGGraph, DAGTask

dag = DAGGraph(tasks=[...])
controller = ParallelSwarmController(
    objective="Build feature",
    config=config,
    repo_root=Path("."),
    max_parallel=4,
    preserve_on_failure=True,  # Keep worktrees for debugging
)
controller.set_dag(dag)

match await controller.run():
    case Ok(merge_result):
        print(f"Merged: {merge_result.merged_branches}")
    case Err(error):
        print(f"Failed: {error}")
```

---

## AST-Aware Context Slicing

Smart code extraction that preserves semantic relationships.

### DependencyWalker

Analyzes Python source to extract:
- **Symbols**: Functions, classes, constants
- **Call Graph**: What calls what
- **Class Hierarchy**: Inheritance relationships
- **Imports**: External dependencies

```python
from jpscripts.core.dependency_walker import DependencyWalker

walker = DependencyWalker(source_code)

# Extract all symbols
symbols = walker.get_symbols()
for s in symbols:
    print(f"{s.kind}: {s.name} ({s.start_line}-{s.end_line})")

# Get call relationships
graph = walker.get_call_graph()
print(graph.callers["main"])  # What main() calls

# Slice with dependencies
context = walker.slice_for_symbol("process_data")

# Fit within token budget
truncated = walker.slice_to_budget("main", max_tokens=500)
```

### Token-Aware Allocation

```python
from jpscripts.core.tokens import TokenBudgetManager, SemanticSlicer

# Priority-based allocation
manager = TokenBudgetManager(total_budget=4000, model="gpt-4o")
content = manager.allocate_with_dependencies(
    priority=1,
    content=full_source,
    target_symbol="main",
)

# Multi-file slicing
slicer = SemanticSlicer()
sliced_files = slicer.prioritize_files(
    files=[path1, path2, path3],
    target_symbols=["foo", "bar"],
    max_tokens=8000,
)
```

---

## Pattern Synthesis

The memory system extracts generalized patterns from successful execution traces.

### How Patterns Are Learned

1. **Trace Analysis**: Reviews last 50 successful trace steps from `~/.jpscripts/traces/`.
2. **Clustering**: Groups similar fixes by error type and solution approach.
3. **LLM Synthesis**: Extracts generalized patterns from clusters with 2+ examples.
4. **Storage**: Patterns stored in dedicated LanceDB collection (`patterns` table).

### Pattern Injection

Relevant patterns are automatically injected into agent prompts:
- Matched by semantic similarity to current task
- Filtered by confidence threshold (60%)
- Provides solution approaches that worked before

### Manual Consolidation

```bash
# Run pattern extraction
jp memory consolidate --model claude-sonnet-4-5

# View learned patterns (via LanceDB directly)
```

### Pattern Structure

- **pattern_type**: `fix_pattern`, `refactor_pattern`, `test_pattern`
- **trigger**: When to apply (error type, code smell, etc.)
- **solution**: Generic solution approach
- **confidence**: 0.0-1.0 based on consistency across examples

---

## Constitutional Governance

All agent-generated code is checked against AGENTS.md constitutional rules.

### Enforcement Strategy: Warn + Prompt

When violations are detected in proposed patches:
1. Violations formatted as structured feedback
2. Agent prompted to revise the patch (single retry)
3. Remaining violations logged for transparency

### Checked Violations

| Type | Rule | Severity |
|:-----|:-----|:---------|
| `SYNC_SUBPROCESS` | `subprocess.run` in async without `asyncio.to_thread` | error |
| `BARE_EXCEPT` | `except:` without specific exception | error |
| `SHELL_TRUE` | `shell=True` in subprocess calls | error |
| `UNTYPED_ANY` | `Any` without `type: ignore` comment | warning |
| `OS_SYSTEM` | `os.system()` usage (always forbidden) | error |
| `SYNC_OPEN` | `open()` in async context without wrapping | warning |

### How It Works

The governance check runs automatically in `AgentEngine.step()`:
1. Parse proposed diff
2. Apply AST analysis to changed Python code
3. Detect constitutional violations
4. If violations found, inject feedback and re-prompt
5. Log remaining violations after retry

---

## Emergency Protocols

- **Agent stuck/unsafe**: Re-run with `--read-only` (no patch apply). Inspect diffs manually.  
- **Repeated failure**: Drop to manual mode: `jp map` + `jp status-all`, open files, patch by hand, then re-run `jp fix` once.  
- **Tool failure**: `jp doctor` to validate binaries; verify `.jpconfig` paths.  
- **Sandbox breach risk**: Manually vet commands; never bypass `validate_command`.  
- **Rollback**: If auto patches misfire, revert affected files (`git checkout -- <paths>` or `jp gundo-last` if ahead-only).  
- **Operational pause**: Stop `jp watch` before destructive refactors; restart after structure stabilizes.

---

## Prompt Engineering Guide (Precision Comms)

- **Architect Persona** (strategy/constraints):  
  - Use when you need plans, guardrails, or multi-step execution.  
  - Speak in objectives and constraints: “Design a 3-step migration for X; forbid data loss; prefer async I/O.”  
  - Ask for risks and rollback steps.

- **Engineer Persona** (surgical edits):  
  - Use for concrete patches and tests.  
  - Provide the failing command, error excerpt, and desired end state.  
  - Specify boundaries: files to touch, patterns to avoid, acceptable refactors.

- **General Rules**:  
  - Supply minimal but exact context (paths, commands, error tail).  
  - Declare priorities (perf vs. correctness vs. speed).  
  - Request diffs, not prose. Demand tests when touching critical paths.

---

## Field Notes (Daily Discipline)

- Keep `jp watch` running; let it block on syntax before you git-add.  
- [Protocol: pre-commit] -> run "jp handbook internal-update-reference"  
- [Protocol: pre-commit] -> run "jp status-all"  
- Capture lessons with `jp memory add ... --tag <area>`; this feeds immediate recall.  
- When handing off, run `jp serialize snapshot` to produce a lossless manifest for teammates or agents.

## Reference

For the complete CLI commands and MCP tools reference, see [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md).

**Quick Help:**
- `jp com` — List all available commands
- `jp <command> --help` — Detailed usage for any command
