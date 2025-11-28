# jpscripts

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The modern, typed, "God-Mode" CLI for macOS power users.**

For detailed workflows and God-Mode configurations, see the [Handbook](HANDBOOK.md).

## Installation

### Prerequisites

- `brew install fzf ripgrep git`
- `brew install gh` (for PR helpers)
- `brew install zoxide` (for navigation)
- Optional: `brew install switchaudio-osx` (for audio control)

### Install with pipx

```bash
pipx install git+https://github.com/jpsweeney97/jp-scripts.git
```

### Development setup

```bash
git clone https://github.com/jpsweeney97/jp-scripts.git
cd jp-scripts
pip install -e ".[dev]"
# Install with AI memory support
pip install -e ".[ai]"
```

## Config Reference

`jpscripts` loads configuration in this order: CLI flags → environment variables → config file (`~/.jpconfig` or `JPSCRIPTS_CONFIG`) → defaults. Run `jp init` to generate a starter file.

Example TOML:

```toml
editor = "code -w"                 # default editor
notes_dir = "~/Notes/quick-notes"  # daily notes location
workspace_root = "~/Projects"      # base for jp recent / proj
ignore_dirs = [".git", "node_modules", ".venv", "__pycache__", "dist", "build", ".idea", ".vscode"]
snapshots_dir = "~/snapshots"
log_level = "INFO"
default_model = "gpt-5.1-codex"
memory_store = "~/.jp_memory.jsonl"
memory_model = "all-MiniLM-L6-v2"
use_semantic_search = true
max_file_context_chars = 50000
max_command_output_chars = 20000
```

## CLI Reference

### Git Operations

| Command              | Description                                               | Key Flags                         |
| :------------------- | :-------------------------------------------------------- | :-------------------------------- |
| `jp status-all`      | Dashboard of all repos (ahead/behind/dirty).              | `--root`, `--max-depth`           |
| `jp whatpush`        | Show outgoing commits and diffstat for the current repo.  | `--max-commits`                   |
| `jp gstage`          | Interactively stage files using `fzf`.                    | `--no-fzf`                        |
| `jp gundo-last`      | Safely undo the last commit (refuses if behind upstream). | `--hard`                          |
| `jp gpr`             | View, checkout, or copy GitHub PR URLs.                   | `--action [view\|checkout\|copy]` |
| `jp gbrowse`         | Open the current repo/branch/commit in GitHub.            | `--target [repo\|commit\|branch]` |
| `jp stashview`       | Browse, apply, pop, or drop git stashes interactively.    | `--action [apply\|pop\|drop]`     |
| `jp git-branchcheck` | List all local branches with upstream tracking status.    |                                   |

### Navigation & Search

| Command            | Description                                          | Key Flags                        |
| :----------------- | :--------------------------------------------------- | :------------------------------- |
| `jp recent`        | Fuzzy-find recently modified files or directories.   | `--files-only`, `--include-dirs` |
| `jp proj`          | Jump to frequently accessed projects (via `zoxide`). |                                  |
| `jp ripper`        | Interactive code search using `ripgrep` + `fzf`.     | `--context`                      |
| `jp todo-scan`     | Scan for TODO/FIXME/HACK/BUG markers.                | `--types`                        |
| `jp loggrep`       | Search or tail log files with regex support.         | `--follow`                       |
| `jp brew-explorer` | Fuzzy-search Homebrew formulas and casks.            | `--query`                        |

### System Utilities

| Command           | Description                                                  | Key Flags         |
| :---------------- | :----------------------------------------------------------- | :---------------- |
| `jp process-kill` | Kill processes by name or port.                              | `--force`         |
| `jp port-kill`    | Kill the process listening on a specific port.               | `--force`         |
| `jp audioswap`    | Switch macOS audio output source.                            |                   |
| `jp ssh-open`     | Fuzzy-select SSH hosts from `~/.ssh/config`.                 |                   |
| `jp tmpserver`    | Start a simple Python HTTP server in the current dir.        | `--port`, `--dir` |
| `jp doctor`       | Verify all external dependencies (`git`, `fzf`, `gh`, etc.). | `--tool`          |

### Notes & Productivity

| Command           | Description                                                  | Key Flags                    |
| :---------------- | :----------------------------------------------------------- | :--------------------------- |
| `jp note`         | Append to today's daily note (or open it).                   | `--message`                  |
| `jp note-search`  | Grep through daily notes with preview.                       | `--no-fzf`                   |
| `jp standup`      | Aggregate commits across all repos for the last N days.      | `--days`                     |
| `jp standup-note` | Run `standup` and append to today's note.                    | `--days`                     |
| `jp cliphist`     | Manage clipboard history backed by SQLite (no corruption).   | `--action [add\|pick\|show]` |
| `jp web-snap`     | Scrape a URL to YAML for LLM context.                        |                              |
| `jp repo-map`     | Pack repo files into XML for LLM paste (respects .gitignore) | `--max-lines`                |

### Agents

| Command         | Description                                                 | Key Flags |
| :-------------- | :---------------------------------------------------------- | :-------- |
| `jp team swarm` | Launch architect/engineer/QA agents in parallel for a task. |           |

### Memory

| Command            | Description                         | Key Flags        |
| :----------------- | :---------------------------------- | :--------------- |
| `jp memory add`    | Append a memory with optional tags. | `--tag` (repeat) |
| `jp memory search` | Search stored memories for a query. | `--limit`        |

### Core

| Command      | Description                                         |
| :----------- | :-------------------------------------------------- |
| `jp init`    | Interactive setup wizard for `~/.jpconfig`.         |
| `jp config`  | View active configuration and source (file vs env). |
| `jp version` | Show version.                                       |
