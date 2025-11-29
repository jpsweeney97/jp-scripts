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
from jpscripts.core.parallel_swarm import ParallelSwarmController
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

## CLI Reference

### CLI Commands
| Command | Args | Description |
| :--- | :--- | :--- |
| `agent` | prompt, --recent/-r, --diff, --run/-x, --full-auto/-y, --model/-m, --provider/-p, --loop, --max-retries, --keep-failed, --archive, --web | Delegate a task to an LLM agent. Supports multiple providers: - Anthropic Claude (claude-opus-4-5, claude-sonnet-4-5, etc.) - OpenAI GPT/o1 (gpt-4o, o1, etc.) - Codex CLI (default for backward compatibility) Examples: jp agent "Fix the failing test" --run "pytest tests/" jp agent "Explain this code" --model claude-opus-4-5 --provider anthropic jp fix "Debug the error" --run "python main.py" --loop |
| `audioswap` | --no-fzf | Switch audio output device using SwitchAudioSource. |
| `brew-explorer` | --query/-q, --no-fzf | Search brew formulas/casks and show info. |
| `cliphist` | --action/-a, --limit/-l, --no-fzf | Simple clipboard history backed by SQLite. |
| `com` | — | Display the available jp commands and their descriptions. |
| `config` | — | Show the active configuration and where it came from. |
| `config-fix` | — | Attempt to fix a broken configuration file using Codex. |
| `doctor` | --tool/-t | Inspect external dependencies in parallel. |
| `fix` | prompt, --recent/-r, --diff, --run/-x, --full-auto/-y, --model/-m, --provider/-p, --loop, --max-retries, --keep-failed, --archive, --web | Delegate a task to an LLM agent. Supports multiple providers: - Anthropic Claude (claude-opus-4-5, claude-sonnet-4-5, etc.) - OpenAI GPT/o1 (gpt-4o, o1, etc.) - Codex CLI (default for backward compatibility) Examples: jp agent "Fix the failing test" --run "pytest tests/" jp agent "Explain this code" --model claude-opus-4-5 --provider anthropic jp fix "Debug the error" --run "python main.py" --loop |
| `gbrowse` | --repo/-r, --target | Open the current repo/branch/commit on GitHub. |
| `git-branchcheck` | --repo/-r | List branches with upstream and ahead/behind counts. |
| `gpr` | --action/-a, --limit, --no-fzf | Interact with GitHub PRs via gh (Typed & Robust). |
| `gstage` | --repo/-r, --no-fzf | Interactively stage files. |
| `gundo-last` | --repo/-r, --hard | Safely undo the last commit. Works on local branches too. |
| `handbook verify-protocol` | --name/-n | Execute Handbook protocol commands for the given context. |
| `init` | --config-path, --install-hooks | Interactive initializer that writes the active config file. |
| `loggrep` | pattern, --path/-p, --no-fzf, --follow/-f | Friendly log search with optional follow mode. |
| `map` | --root/-r, --depth/-d | Generate a concise project structure map with top-level symbols. |
| `memory add` | content, --tag/-t | Add a memory entry. |
| `memory consolidate` | --model/-m, --threshold | Cluster similar memories and synthesize canonical truth entries. |
| `memory reindex` | --force/-f | — |
| `memory search` | query, --limit/-l | Search memory for relevant entries. |
| `memory vacuum` | — | Remove memory entries related to deleted files to maintain vector store hygiene. |
| `note` | --message/-m | Append to today's note or open it in the configured editor. |
| `note-search` | query, --no-fzf | Search notes with ripgrep and optionally fzf. |
| `port-kill` | port, --force/-f, --no-fzf | Find processes bound to a port and kill one. |
| `process-kill` | --name/-n, --port/-p, --force/-f, --no-fzf | Interactively select and kill a process. |
| `proj` | --no-fzf | Fuzzy-pick a project using zoxide + fzf and print the path. |
| `recent` | --root/-r, --limit/-l, --max-depth, --include-dirs, --files-only, --no-fzf | Fuzzy-jump to recently modified files or directories. |
| `repo-map` | --root/-r, --depth/-d | Generate a concise project structure map with top-level symbols. |
| `ripper` | pattern, --path/-p, --no-fzf, --context/-C | Interactive code search using ripgrep + fzf. |
| `serialize snapshot` | --output/-o, --format/-f | — |
| `ssh-open` | --host/-h, --no-fzf | Fuzzy-pick an SSH host from ~/.ssh/config and connect. |
| `standup` | --days/-d, --max-depth | Summarize recent commits across repos. |
| `standup-note` | --days/-d | Run standup and append its output to today's note. |
| `stashview` | --repo/-r, --action/-a, --no-fzf | Browse stash entries and apply/pop/drop one. |
| `status-all` | --root/-r, --max-depth | Summarize git status across repositories with a live-updating table. |
| `sync` | --root/-r, --max-depth | Parallel git fetch across all repositories. |
| `team swarm` | objective | Launch architect, engineer, and QA Codex agents in parallel. |
| `tmpserver` | --dir/-d, --port/-p | Start a simple HTTP server. |
| `todo-scan` | --path/-p, --types | Scan for TODO items and display a structured table. |
| `trace list` | --limit/-n | List recent execution traces. |
| `trace show` | trace_id, --watch/-w | Display detailed trace for a specific execution. |
| `update` | — | Update jpscripts in editable installs, or guide pipx users. |
| `version` | — | Print the jpscripts version. |
| `watch watch` | — | Run a God-Mode file watcher that triggers syntax checks and memory updates. |
| `web-snap` | url | Fetch a webpage, extract main content, and save as a YAML snapshot. |
| `whatpush` | --repo/-r, --max-commits | Show what will be pushed to the upstream branch. |

### MCP Tools
| Tool | Params | Description |
| :--- | :--- | :--- |
| `append_daily_note` | message: str | Append a log entry to the user's daily note system. |
| `apply_patch` | path: str, diff: str | Apply a unified diff to a file within the workspace. Args: path: Target file path, absolute or relative to the workspace root. diff: Unified diff content to apply. Returns: Status message describing whether the patch was applied. |
| `fetch_url_content` | url: str | Fetch and parse a webpage into clean Markdown. |
| `find_todos` | path: str='.' | Scan for TODO/FIXME/HACK comments in the codebase. Returns a JSON list of objects: {type, file, line, text}. |
| `get_git_status` | — | Return a summarized git status. |
| `get_workspace_status` | max_depth: int=2 | Summarize branch status for repositories in the workspace. Args: max_depth: Depth to search for git repositories under workspace_root. Returns: Formatted summary lines containing repo name, branch, and ahead/behind counts. |
| `git_commit` | message: str | Stage all changes and create a commit. |
| `kill_process` | pid: int, force: bool=False | Kill a process by PID. |
| `list_directory` | path: str | List contents of a directory (like ls). Returns a list of 'd: dir_name' and 'f: file_name'. |
| `list_processes` | name_filter: str | None=None, port_filter: int | None=None | List running processes. |
| `list_projects` | — | List known projects (via zoxide). |
| `list_recent_files` | limit: int=20 | List files modified recently in the current workspace root and surface related memories. |
| `read_file` | path: str | Read the content of a file (truncated to JP_MAX_FILE_CONTEXT_CHARS). Use this to inspect code, config files, or logs. |
| `read_file_paged` | path: str, offset: int=0, limit: int=20000 | Read a file segment starting at byte offset. Use this to read large files. |
| `recall` | query: str, limit: int=5 | Retrieve the most relevant memories for a query. |
| `remember` | fact: str, tags: str | None=None | Save a fact or lesson to the persistent memory store. Tags can be provided as a comma-separated list. |
| `run_shell` | command: str | Execute a safe, sandboxed command without shell interpolation. Only allows read-only inspection commands. |
| `run_tests` | target: str='.', verbose: bool=False | Run pytest on a specific target (directory or file) and return the results. Use this to verify fixes. |
| `search_codebase` | pattern: str, path: str='.' | Search the codebase using ripgrep (grep). Returns the raw text matches with line numbers. |
| `write_file` | path: str, content: str, overwrite: bool=False | Create or overwrite a file with the given content. Enforces workspace sandbox. Requires overwrite=True to replace existing files. |
