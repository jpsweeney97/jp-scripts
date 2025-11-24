# jpscripts

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The modern, typed, "God-Mode" CLI for macOS power users.**

`jpscripts` is a complete rewrite of legacy Bash tooling into a unified Python application powered by **Typer**, **Rich**, and **GitPython**. It provides high-leverage utilities for git repository management, deep system navigation, productivity logging, and AI-context generation.

## Prerequisites

`jpscripts` orchestrates powerful system binaries. You must have these installed:

```bash
# Core dependencies (Required)
brew install fzf ripgrep git

# Git Extensions (Required for 'jp gpr', 'jp gbrowse')
brew install gh

# Navigation (Required for 'jp proj')
brew install zoxide

# Optional: Audio control
brew install switchaudio-osx
```

## Installation

Recommended installation via `pipx` for isolation:

```bash
pipx install git+[https://github.com/jpsweeney97/jp-scripts.git](https://github.com/jpsweeney97/jp-scripts.git)
```

_Development setup:_

```bash
git clone [https://github.com/jpsweeney97/jp-scripts.git](https://github.com/jpsweeney97/jp-scripts.git)
cd jp-scripts
pip install -e ".[dev]"
```

## Command Reference

### Git Operations

_Manage your repositories without leaving the terminal._

| Command              | Description                                                    | Key Flags                         |
| :------------------- | :------------------------------------------------------------- | :-------------------------------- |
| `jp status-all`      | **Daily Driver.** Dashboard of all repos (ahead/behind/dirty). | `--root`, `--max-depth`           |
| `jp whatpush`        | Show outgoing commits and diffstat for the current repo.       | `--max-commits`                   |
| `jp gstage`          | Interactively stage files using `fzf`.                         | `--no-fzf`                        |
| `jp gundo-last`      | Safely undo the last commit (refuses if behind upstream).      | `--hard`                          |
| `jp gpr`             | View, checkout, or copy GitHub PR URLs.                        | `--action [view\|checkout\|copy]` |
| `jp gbrowse`         | Open the current repo/branch/commit in GitHub.                 | `--target [repo\|commit\|branch]` |
| `jp stashview`       | Browse, apply, pop, or drop git stashes interactively.         | `--action [apply\|pop\|drop]`     |
| `jp git-branchcheck` | List all local branches with upstream tracking status.         |                                   |

### Navigation & Search

_Move fast, find things faster._

| Command            | Description                                          | Key Flags                        |
| :----------------- | :--------------------------------------------------- | :------------------------------- |
| `jp recent`        | Fuzzy-find recently modified files or directories.   | `--files-only`, `--include-dirs` |
| `jp proj`          | Jump to frequently accessed projects (via `zoxide`). |                                  |
| `jp ripper`        | Interactive code search using `ripgrep` + `fzf`.     | `--context`                      |
| `jp todo-scan`     | Scan for TODO/FIXME/HACK/BUG markers.                | `--types`                        |
| `jp loggrep`       | Search or tail log files with regex support.         | `--follow`                       |
| `jp brew-explorer` | Fuzzy-search Homebrew formulas and casks.            | `--query`                        |

### System Utilities

_Control your machine._

| Command           | Description                                                  | Key Flags         |
| :---------------- | :----------------------------------------------------------- | :---------------- |
| `jp process-kill` | Kill processes by name or port.                              | `--force`         |
| `jp port-kill`    | Kill the process listening on a specific port.               | `--force`         |
| `jp audioswap`    | Switch macOS audio output source.                            |                   |
| `jp ssh-open`     | Fuzzy-select SSH hosts from `~/.ssh/config`.                 |                   |
| `jp tmpserver`    | Start a simple Python HTTP server in the current dir.        | `--port`, `--dir` |
| `jp doctor`       | Verify all external dependencies (`git`, `fzf`, `gh`, etc.). | `--tool`          |

### Notes & Productivity

_Capture thoughts and context._

| Command           | Description                                                | Key Flags                    |
| :---------------- | :--------------------------------------------------------- | :--------------------------- |
| `jp note`         | Append to today's daily note (or open it).                 | `--message`                  |
| `jp note-search`  | Grep through daily notes with preview.                     | `--no-fzf`                   |
| `jp standup`      | Aggregate commits across all repos for the last N days.    | `--days`                     |
| `jp standup-note` | Run `standup` and append to today's note.                  | `--days`                     |
| `jp cliphist`     | Manage clipboard history backed by SQLite (no corruption). | `--action [add\|pick\|show]` |
| `jp web-snap`     | **Context Fetcher.** Scrape a URL to YAML for LLM context. |                              |

### Core

| Command      | Description                                         |
| :----------- | :-------------------------------------------------- |
| `jp init`    | Interactive setup wizard for `~/.jpconfig`.         |
| `jp config`  | View active configuration and source (file vs env). |
| `jp version` | Show version.                                       |

## Configuration

`jpscripts` loads configuration in the following precedence order:

1.  **CLI Flags** (e.g., `--root`)
2.  **Environment Variables** (e.g., `JP_NOTES_DIR`)
3.  **Config File** (`~/.jpconfig` or `JPSCRIPTS_CONFIG`)
4.  **Defaults**

### TOML Configuration (`~/.jpconfig`)

Run `jp init` to generate this file interactively.

```toml
# The command to open files (e.g., from 'jp note')
editor = "code -w"

# Where 'jp note' and 'jp standup-note' save files
notes_dir = "~/Notes/quick-notes"

# Root for 'jp recent' scans
workspace_root = "~/Projects"

# Directory names to ignore when scanning (default: [".git", "node_modules", ".venv", "__pycache__", "dist", "build", ".idea", ".vscode"])
ignore_dirs = [".git", "node_modules", ".venv", "__pycache__", "dist", "build", ".idea", ".vscode"]

# Where 'jp web-snap' saves YAML snapshots
snapshots_dir = "~/snapshots"

# Global log level (DEBUG, INFO, WARNING, ERROR)
log_level = "INFO"
```

## Codex & Agent Integration

This repository includes an `AGENTS.md` file specifically designed to steer OpenAI's Codex.

- **Project Structure**: Defines where commands live (`src/jpscripts/commands`) vs core logic.
- **Testing**: Instructs agents to use `pytest tests/test_smoke.py` for verification.
- **Conventions**: Enforces Typer/Rich usage for all new commands.

**To create a new command with Codex:**

> "Read AGENTS.md. Create a new command `jp network-scan` in `src/jpscripts/commands/system.py` that uses `lsof` to find listening ports. Follow the existing patterns for `process-kill`."

## License

MIT
