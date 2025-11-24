# jpscripts

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The modern, typed, "God-Mode" CLI for macOS power users.**

`jpscripts` is a complete rewrite of legacy Bash tooling into a unified Python application powered by **Typer**, **Rich**, and **GitPython**. It provides high-leverage utilities for git repository management, deep system navigation, productivity logging, and AI-context generation.

## Prerequisites

While `jpscripts` is a Python package, it orchestrates powerful system binaries. You must have these installed for full functionality:

```bash
# Required for core navigation and search
brew install fzf ripgrep

# Required for 'jp proj' and git interactions
brew install zoxide gh git

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

### AI & Context Gathering

Tools designed to feed context to LLMs (like Codex) or archive knowledge.

| Command                  | Description                                                                                                                                                                  | Key Flags  |
| :----------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------- |
| `jp web-snap <url>`      | **The Context Fetcher.** Scrapes a URL, extracts main content (removing boilerplate), and saves it as a structured YAML snapshot. Perfect for feeding documentation to LLMs. |            |
| `jp note-search <query>` | Greps through your daily notes with `ripgrep` and creates an interactive `fzf` selection menu.                                                                               | `--no-fzf` |

### Git Operations

Stop manually checking 50 repositories.

| Command         | Description                                                                                                                                       | Key Flags                 |
| :-------------- | :------------------------------------------------------------------------------------------------------------------------------------------------ | :------------------------ |
| `jp status-all` | **The Daily Driver.** Scans your workspace recursively and displays a live-updating dashboard of every repo's status (ahead/behind/dirty/staged). | `--max-depth`, `--root`   |
| `jp whatpush`   | Shows exactly what commits will be sent to upstream. Includes a diffstat summary.                                                                 | `--max-commits`           |
| `jp gbrowse`    | Opens the current branch, commit, or file in GitHub. Logic handles SSH/HTTPS remotes automatically.                                               | `--target [repo\|commit]` |
| `jp gundo-last` | Safely undoes the last commit (`reset --soft` by default). **Safety check:** Refuses to run if you are behind upstream.                           | `--hard`                  |

### Navigation & System

Fast movement through large monorepos.

| Command           | Description                                                                                               | Key Flags                 |
| :---------------- | :-------------------------------------------------------------------------------------------------------- | :------------------------ |
| `jp recent`       | Fuzzy-find recently modified files. Intelligently ignores noise like `node_modules`, `.venv`, and `.git`. | `--files-only`, `--limit` |
| `jp proj`         | A wrapper around `zoxide` to fuzzy-jump to frequently accessed projects.                                  |                           |
| `jp process-kill` | Interactive process killer. Filters by name or port, shows cmdline arguments.                             | `--port`, `--force`       |
| `jp audioswap`    | Instantly fuzzy-select and switch macOS audio output devices (requires `SwitchAudioSource`).              |                           |

### Productivity

| Command           | Description                                                                  |
| :---------------- | :--------------------------------------------------------------------------- |
| `jp standup`      | Aggregates your commits across _all_ local repositories for the last N days. |
| `jp standup-note` | Runs `standup` and appends the report directly to today's daily note.        |
| `jp init`         | Interactive bootstrap wizard to generate your `~/.jpconfig`.                 |

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
