# Changelog

## [0.4.4] - Unreleased

### Added
- Integration safeguard for Codex handoff with a mocked subprocess and XML prompt validation (`tests/integration/test_agent_real.py`).
- Async git core helpers for remote URL lookup, stash management, and porcelain-short status parsing to replace GitPython-only surfaces.
- `make lint` entry for strict `mypy` gating ahead of tests.

### Changed
- Git plumbing now uses true asyncio subprocess calls and porcelain v2 parsing for status, fetch, and commit workflows (no GitPython threads).
- `git-extra` commands (`gstage`, `gbrowse`, `stashview`) now run through `AsyncRepo` with asyncio orchestration, removing direct GitPython dependencies.
- Python context reads fall back to warn-tagged head/tail slices when AST parsing fails, preserving context through syntax errors.
- Memory storage migrated to LanceDB with a pydantic schema and vector search, while retaining keyword fallback when AI extras are absent.

## [0.4.0] - The Trinity Update

### Added
- **Architecture Map**: New `jp map` command (and `repo-map` alias) generates high-density AST summaries of the codebase for context-efficient planning.
- **Diff-Aware Agents**: `jp fix` now automatically attaches `git diff` context so agents can see uncommitted changes.

### Changed
- **Modular MCP**: Refactored monolithic `mcp_server.py` into a modular `jpscripts.mcp` package with dynamic tool registration.
- **Documentation**: Updated `AGENTS.md` to mandate `jp map` usage for architectural exploration.

## [0.3.0] - 2025-11-25

### Security Hardening

- Enforced workspace-root sandbox for MCP file tools via `validate_path`, rejecting traversal and symlink escapes.
- Hard-capped context reads at 100KB to prevent prompt/context DOS attacks in MCP and agent flows.

### Refactoring

- Restored strict Core/Command separation (UI helpers moved to commands, agent orchestration lives in core, git ops consolidated).
- Git utilities now share core helpers for undo/branch status, keeping command surfaces thin and reusable.

## [0.2.1] - November 25th Update

### Added

- **Semantic Memory:** Local, offline embedding engine (MPS-accelerated) for `jp memory search`.
- **Safety Core:** Unified context limiting (`JP_MAX_FILE_CONTEXT_CHARS`) and streaming I/O to prevent OOM.
- **MCP Hardening:** Full tool exposure (`read_file`, `search_codebase`) with enforced safety truncation.

### Changed

- **Performance:** Lazy loading of heavy AI libraries (torch/numpy) keeps CLI startup under 100ms.
- **Git Ops:** Optimized `status-all` commit counting to O(1) using raw plumbing commands.

## [0.2.0] - The Stability Update

### Changed

- `cliphist` now uses a SQLite database (`history.db`) for atomic, corruption-free writes.
- `jp doctor` install hints are now platform-agnostic.

### Added

- New `ignore_dirs` configuration option in `~/.jpconfig` to override default ignored directories in `jp nav`.
- Comprehensive unit testing infrastructure (`tests/unit/`) and a project `Makefile` for standardized workflows (`install`, `test`, `format`).

### Fixed

- Reliable macOS detection in `jp web-snap` using `sys.platform`.

## [0.1.0] - Python Rewrite

**The Modern Era.**

`jpscripts` has been completely rewritten from legacy Bash scripts into a unified Python CLI using **Typer** and **Rich**.

### Added

- **Core:** Unified entry point `jp` with subcommands.
- **Config:** TOML-based configuration (`~/.jpconfig`) with environment variable overrides.
- **Git:** `status-all` (async), `whatpush`, `gstage`, `gundo-last`.
- **Nav:** `recent` (smart mtime sorting), `proj` (zoxide wrapper).
- **System:** `process-kill` (interactive filtering), `audioswap`, `tmpserver`.
- **AI:** `web-snap` for turning URLs into LLM-ready YAML contexts.
- **Notes:** Daily note management (`note`, `note-search`) and automated standups.

### Removed

- All legacy `bin/*.sh` scripts.
- `lib/` shell helpers (replaced by `jpscripts.core`).
