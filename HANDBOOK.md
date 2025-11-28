# The JPScripts Handbook: Operating in God-Mode

Welcome to `jpscripts`. This isn't just a collection of shell aliases; it is a coherent, typed, AI-native operating system for your terminal.

This handbook describes **workflows**, not just commands. It assumes you want to move faster than your hardware allows, leveraging local LLMs, semantic memory, and aggressive automation.

---

## 1. The Philosophy: Context is King

Most CLIs are stateless. `jpscripts` is different—it maintains **context** about what you are doing, where you have been, and what you have learned.

- **Spatial Context**: `jp recent` and `jp proj` know where you work.
- **Temporal Context**: `jp standup` and `jp note` know what you did.
- **Semantic Context**: `jp memory` and `jp team` know _how_ you solve problems.

---

## 2. The Daily Loop

Do not treat `jp` commands as isolated utilities. Chain them into a daily loop.

### 09:00 — The "Download"

Start your day by recalling exactly where you left off without opening a browser or checking Slack.

1.  **Aggregate Context**:

    ```bash
    # View a dashboard of every repo you touched in the last 3 days
    jp standup --days 3

    # Check the "health" of your workspace (ahead/behind status)
    jp status-all
    ```

2.  **Hydrate the Daily Note**:
    Instead of typing your status manually, generate it.
    ```bash
    # Runs the standup scan and appends the summary to ~/Notes/quick-notes/YYYY-MM-DD.md
    jp standup-note
    ```

### 10:00 — Deep Work (Navigation)

Stop `cd`-ing into directories. Stop running `ls`.

- **Jump to Project**: `jp proj` (wraps `zoxide` to fuzzy-find repositories).
- **Jump to File**: `jp recent` (sorts by _mtime_, so the file you just edited is always at the top).
- **Context Switch**: `jp recent --files-only` allows you to immediately open the specific file you were working on yesterday.

### 14:00 — The "Fix It" Loop (AI Augmented)

This is the core "God-Mode" workflow. When a test fails or a bug appears, do not copy-paste logs into a web browser.

**The `jp fix` Strategy:**
The `jp fix` command (preferred; `jp agent` is a legacy alias) supports **Just-In-Time (JIT) RAG**. It runs a command, captures the stdout/stderr, finds file paths in the stack trace, reads those files, and constructs a prompt for the LLM.

**Multi-Provider Support:**
`jp fix` works with multiple LLM providers:
- **Anthropic Claude** (`claude-opus-4-5`, `claude-sonnet-4-5`, etc.)
- **OpenAI GPT/o1** (`gpt-4o`, `o1`, `o1-mini`, etc.)
- **Codex CLI** (default for OpenAI models when installed)

```bash
# Use default provider (auto-detected from model)
jp fix "Explain this error" --run "pytest tests/"

# Explicitly use Anthropic Claude
jp fix "Debug this" --model claude-opus-4-5 --provider anthropic

# Explicitly use OpenAI
jp fix "Fix the type error" --model gpt-4o --provider openai

# Force legacy Codex CLI path
jp fix "Refactor this" --legacy-codex
```

- **Scenario**: Unit tests are failing.

  ```bash
  # 1. Run the failing test via the agent
  # 2. Agent sees the stack trace pointing to src/core/auth.py:42
  # 3. Agent reads src/core/auth.py
  # 4. Agent proposes a fix
  jp fix --run "pytest tests/unit/test_auth.py" "The tests are failing. Fix the off-by-one error."
  ```

- **Scenario**: "I broke something 10 minutes ago."
  ```bash
  # Attach the top 5 most recently modified files as context automatically
  jp fix --recent "I broke the build. Look at what I just changed and fix it."
  ```

### 17:00 — Archival

Before you sign off, dump your brain to disk.

1.  **Semantic Memory**: Did you learn a tricky CLI flag or an architectural decision?
    ```bash
    jp memory add "We use uv instead of poetry because poetry locking is too slow on CI" --tag infra
    ```
2.  **Clip History**: Don't lose that code snippet you copied 3 hours ago.
    ```bash
    jp cliphist --action pick
    ```

---

## 3. Advanced Git Operations

Git is the database of your work. `jpscripts` provides a specialized query engine for it.

### The "Safety Net" (`jp gundo-last`)

We all commit to the wrong branch or make typos. `jp gundo-last` performs a **safe** reset.

- It checks if you are `ahead` of upstream.
- If you are ahead (local commits only), it effectively "pops" the commit back to staging.
- If you are behind (pushed commits), it **refuses** to run to prevent rewriting shared history.

### The "Pull Request" Dashboard (`jp gpr`)

Stop clicking around GitHub.

```bash
# Interactive fzf list of all open PRs
# Press Enter to view in browser
jp gpr

# Checkout a PR locally by selecting it
jp gpr --action checkout
```

### The "Branch Audit" (`jp git-branchcheck`)

Find out which local branches have drifted from `origin/main` or are orphaned.

```bash
jp git-branchcheck
# Output:
# feature/login  origin/feature/login  0/5  (Behind by 5!)
# fix/typo       (none)                0/0  (Not tracking anything)
```

---

## 4. Multi-Agent Swarms

For tasks too complex for a single prompt (e.g., "Refactor this module" or "Add a new feature"), use the Swarm.

```bash
jp team swarm "Refactor the search module to use async ripgrep calls."
```

**How it works:**

1.  **Architect**: Analyzes the request and `git status`. producing a step-by-step plan.
2.  **Engineer**: Takes the plan and generates code blocks.
3.  **QA**: Reviews the Engineer's output against the Architect's plan and invariants.

**Pro Tip:** Swarms run in **Safe Mode** by default. To allow them to execute destructive commands without confirmation, ensure your `~/.codex/config.toml` allows it, though `jp team` is designed to be a "Human-in-the-Loop" planner.

---

## 5. Architecture Mapping

Large refactors benefit from a structural overview. Generate an AST-driven map and feed it straight into the agent loop:

```bash
# Create a repo map and pipe it into jp fix as context
jp map --depth 3 --format text | jp fix --run "cat > /tmp/repo-map.txt" "Use the repo map above to plan a safe refactor."
```

`jp fix` is the primary entrypoint for agents; `jp agent` remains as a legacy alias for older scripts.

---

## 6. The MCP Server (Integration with external Agents)

`jpscripts` is not just a tool; it is a **Server**. You can expose your local tools (file reading, searching, memory) to external AI agents (like Claude Desktop or generic Codex instances).

**Enable God-Mode for Claude:**
Add this to your Claude Desktop config:

```json
{
  "mcpServers": {
    "jpscripts": {
      "command": "uv",
      "args": ["run", "jpscripts.mcp.server"]
    }
  }
}
```

**What this unlocks:**

- **"Read my mind"**: The agent can call `list_recent_files` to see what you are working on _right now_.
- **"Remember this"**: The agent can call `remember` to store facts in your local vector database.
- **"Search everything"**: The agent can use `search_codebase` (ripgrep) to answer questions about your entire project.

---

## 7. Configuration & Troubleshooting

### The `~/.jpconfig`

This TOML file controls the behavior. Run `jp init` to generate it.

**Key Settings:**

- `workspace_root`: The base path for `jp recent` scans. Set this to your code folder (e.g., `~/src`).
- `memory_model`: Defaults to `all-MiniLM-L6-v2` for local embeddings. If you are on a slow machine, disable `use_semantic_search`.

### Self-Healing (`jp config-fix`)

If you mess up your TOML syntax, do not edit it manually.

```bash
# Uses Codex to read the traceback and patch the file automatically
jp config-fix
```

### System Diagnostics (`jp doctor`)

If a tool fails, run the doctor. It checks the $PATH and version compatibility for:

- `git`, `gh`
- `fzf`, `rg` (ripgrep)
- `codex`
- `SwitchAudioSource` (macOS only)

### Security & Async Invariants (do not break)

- **No shell=True ever**: All subprocesses use `asyncio.create_subprocess_exec` with tokenized args; guardrails fail the build if `create_subprocess_shell` or blocking `subprocess.run`/`Popen` reappear in commands (only `commands/ui.py` encapsulates fzf).
- **Async-first I/O**: Keep editors, ssh, fzf, and ripgrep flows behind async helpers or `asyncio.to_thread`; avoid blocking the loop.
- **Workspace sandbox**: Run paths through `security.validate_path`; MCP tools must refuse anything outside `workspace_root`.
- **MCP strictness**: Every MCP tool uses `@tool_error_handler` plus Pydantic validation; tool definitions must mirror AgentEngine tools to stay in sync.

---

## 8. Internal Architecture

This section documents internal design patterns for contributors and advanced users.

### LLM Provider Abstraction

The `jpscripts.providers` module provides a unified interface to multiple LLM backends:

```
┌─────────────────────────────────────────────────────────────┐
│                    LLMProvider Protocol                     │
├─────────────────────────────────────────────────────────────┤
│  complete(messages, model, options) -> CompletionResponse   │
│  stream(messages, model, options) -> AsyncIterator[Chunk]   │
│  supports_streaming() -> bool                               │
│  supports_tools() -> bool                                   │
└─────────────────────────────────────────────────────────────┘
         ▲                    ▲                    ▲
         │                    │                    │
┌────────┴────────┐ ┌────────┴────────┐ ┌────────┴────────┐
│ AnthropicProvider│ │  OpenAIProvider │ │  CodexProvider  │
│ claude-opus-4-5  │ │  gpt-4o, o1     │ │  (CLI wrapper)  │
│ claude-sonnet-4  │ │  gpt-4-turbo    │ │                 │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

**Provider Selection:**
- Auto-detected from model ID prefix (`claude-*` → Anthropic, `gpt-*`/`o1*` → OpenAI)
- Explicitly via `--provider` flag
- Codex CLI used automatically for OpenAI models when installed

**Environment Variables:**
- `ANTHROPIC_API_KEY`: Required for Anthropic provider
- `OPENAI_API_KEY`: Required for OpenAI provider

### Command Validation

Shell commands executed via agents use tokenized validation (`core/command_validation.py`):

1. **Allowlist-based**: Only read-only commands permitted (`cat`, `ls`, `git status`, `rg`, `head`, `tail`, etc.)
2. **No metacharacters**: Pipes, redirects, and shell expansion are rejected
3. **Path sandboxing**: All paths validated against workspace root
4. **Dangerous flag detection**: Flags like `--force`, `-rf`, `--delete` are blocked

```python
# Security verdicts
CommandVerdict.ALLOWED           # Safe to execute
CommandVerdict.BLOCKED_FORBIDDEN # Explicitly blocked command
CommandVerdict.BLOCKED_PATH_ESCAPE # Path traversal attempt
CommandVerdict.BLOCKED_DANGEROUS_FLAG # Risky flag detected
CommandVerdict.BLOCKED_METACHAR  # Shell metacharacters detected
```

### Runtime Context

Thread-safe context management using `contextvars` (`core/runtime.py`):

```python
from jpscripts.core.runtime import get_runtime

ctx = get_runtime()
print(ctx.workspace_root)  # Current workspace
print(ctx.trace_id)        # Request correlation ID
print(ctx.dry_run)         # Dry-run mode flag
```

### Result Types

The codebase uses explicit Result types for error handling (`core/result.py`):

```python
from jpscripts.core.result import Ok, Err, Result

def divide(a: int, b: int) -> Result[float, str]:
    if b == 0:
        return Err("Division by zero")
    return Ok(a / b)

match divide(10, 2):
    case Ok(value):
        print(f"Result: {value}")
    case Err(error):
        print(f"Error: {error}")
```

Functions returning `Result` never raise exceptions for expected errors. Use `is_ok()`, `is_err()`, `unwrap()`, or pattern matching to handle results.

---

## 9. Keyboard Shortcuts (Recommended)

To truly achieve God-Mode, bind these commands to shell aliases or tmux keys:

| Alias | Command           | Mnemonic          |
| :---- | :---------------- | :---------------- |
| `jn`  | `jp note`         | **J**p **N**ote   |
| `jr`  | `jp recent`       | **J**p **R**ecent |
| `jf`  | `jp fix --recent` | **J**p **F**ix    |
| `jg`  | `jp status-all`   | **J**p **G**it    |
| `js`  | `jp ripper`       | **J**p **S**earch |

> _"The tool that is closest to your hand is the one you will use."_
