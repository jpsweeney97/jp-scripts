# AGENTS.md

> **Role**: Principal System Engineer & CLI Architect.
> **Objective**: Maintain `jpscripts` as the "God-Mode" interface for macOS.

## 1. Architectural Invariants

- **The Core/Command Barrier**:

  - `commands/` modules handles I/O, CLI arguments (Typer), and UI rendering (Rich).
  - `core/` modules contain pure logic, data classes, and subprocess/git wrappers.
  - **Violation**: A command module importing another command module.

- **Performance Contracts**:
  - `jp recent` and `jp nav` must resolve in <100ms.
  - Heavy Git operations (`fetch`, `status-all`) must use `asyncio` and provide a `rich.live.Live` spinner.

## 2. Codex Interaction Protocol

When using `@codex` or `jp fix`:

1.  **Smart Context**: Prefer `jp fix -x "pytest" "Fix tests"` over blindly attaching recent files. This grabs the exact files involved in the stack trace.
2.  **Safety**:
    - NEVER write `subprocess.run("rm -rf ...")` without an explicit `typer.confirm`.
    - Git operations must check `git_core.describe_status(repo)` to ensure no uncommitted changes are clobbered.
3.  **MCP Usage**:
    - **Perception**: Use `list_directory`, `read_file`, and `search_codebase` (grep) to investigate the environment before acting.
    - **Knowledge**: Use `fetch_url_content` to read external documentation.
    - **Action**: Use `append_daily_note` to log architectural decisions.

## 3. High-Leverage Refactoring Targets

- **`search.py`**: Convert `todo_scan` to use the `mcp_server` logic to allow agents to auto-fix found TODOs.
- **`system.py`**: `brew_explorer` invokes `brew` synchronously. Refactor to `asyncio.create_subprocess_exec` to prevent UI blocking during network hangs.

## 4. Memory Protocol

- **Retrieval**: Before answering complex questions about the user's preferences or infrastructure, ALWAYS run `jp memory search "<concept>"`.
- **Storage**: When the user makes a decision (e.g., "We are switching to uv for package management"), ALWAYS run `jp memory add "We use uv for package management" --tag infrastructure`.
