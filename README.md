# jpscripts

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![mypy: strict](https://img.shields.io/badge/mypy-strict-blue.svg)](https://mypy.readthedocs.io/)

**The modern, typed, "God-Mode" CLI for macOS power users with autonomous agent orchestration.**

For detailed workflows and God-Mode configurations, see the [Handbook](HANDBOOK.md).

## Highlights

- **Parallel Swarm Execution**: Run multiple AI agents in parallel using git worktrees for isolation
- **AST-Aware Context Slicing**: Smart code slicing that preserves semantic meaning
- **Constitutional Governance**: Automated enforcement of coding standards via AST analysis
- **Merge Conflict Resolution**: Intelligent 3-tier conflict resolution (TRIVIAL → SEMANTIC → COMPLEX)
- **CircuitBreaker Guardrails**: Enforces cost velocity and file churn limits, emitting Black Box crash reports before damage spreads
- **Cognitive Separation**: Captures raw `<thinking>` streams separately from action JSON so reasoning cannot be rewritten by summarization

## Architecture

The system uses a layered architecture with swarm planning, isolated execution, and governance-enforced safety rails.

For detailed diagrams and module descriptions, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Installation

### Prerequisites

- Python 3.11+ (uses `asyncio.TaskGroup` and modern type syntax)
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

See [CONTRIBUTING.md](CONTRIBUTING.md) for the complete development guide.

## Quick Start

```bash
# Verify installation
jp doctor

# Generate config file
jp init

# View your config
jp config
```

## Key Features

### Autonomous Evolution

`jp evolve` proactively identifies and refactors high technical debt code. See the [Handbook](HANDBOOK.md#the-evolve-loop-jp-evolve) for the full protocol.

```bash
jp evolve run --dry-run    # Analyze without changes
jp evolve report           # Show complexity report
jp evolve debt             # Show debt scores
```

### Parallel Swarm Architecture

Execute DAG-based tasks in parallel with full git worktree isolation. See the [Handbook](HANDBOOK.md#parallel-swarm-execution) for usage patterns.

### AST-Aware Context Slicing

Smart code extraction that preserves semantic relationships using the `DependencyWalker`. See the [Handbook](HANDBOOK.md#ast-aware-context-slicing) for examples.

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

## Essential Commands

| Command | Description |
| :--- | :--- |
| `jp doctor` | Verify external dependencies |
| `jp init` | Generate config file |
| `jp config` | Show active configuration |
| `jp agent` / `jp fix` | Delegate task to LLM agent |
| `jp recent` | Jump to recently modified files |
| `jp map` | Generate project structure map |
| `jp status-all` | Git status across all repos |
| `jp memory search` | Query the memory store |
| `jp evolve run` | Autonomous code optimization |
| `jp watch` | God-Mode file watcher |

For the complete CLI and MCP tools reference, see [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md).

## Troubleshooting

- `jp doctor` shows LanceDB/embedding failures: install AI extras (`pip install "jpscripts[ai]"`) or set `use_semantic_search = false` in `~/.jpconfig` if you want JSONL-only mode. Capability errors surface as `CapabilityMissingError` rather than silent degradation.
- `jp doctor` reports MCP config missing: ensure `~/.codex/config.toml` exists or pass `--tool mcp` after running `jp init` to regenerate the file; server discovery requires that config path.
