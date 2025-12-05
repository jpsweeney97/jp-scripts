# Changelog

## [0.9.2](https://github.com/jpsweeney97/jp-scripts/compare/v0.9.1...v0.9.2) (2025-12-05)


### Features

* add NO_OP storage mode and memory_mode config ([d9e54cc](https://github.com/jpsweeney97/jp-scripts/commit/d9e54cca2e97e390165cd03d13ffca57930d3a31))
* Phase 3 - Provider & Agent Interface Formalization ([062e96e](https://github.com/jpsweeney97/jp-scripts/commit/062e96e2c9f1ddcf7283545caf01bae4f7dad7f5))
* Phase 4 - Safety & Concurrency Hardening ([519b6eb](https://github.com/jpsweeney97/jp-scripts/commit/519b6ebe7c803eb5dfcd407b4c443ff33a800766))


### Bug Fixes

* resolve audit roadmap P0-P3 items ([1f6debc](https://github.com/jpsweeney97/jp-scripts/commit/1f6debc06b4d67027a23db4e00910e871963747c))


### Refactoring

* consolidate security into core/security/ package (Phase 1+2) ([9dd5a9b](https://github.com/jpsweeney97/jp-scripts/commit/9dd5a9bd40377f38b16cef895e9697170c7fbf80))
* decompose agent/execution.py into focused modules (Phase 3) ([c837b0b](https://github.com/jpsweeney97/jp-scripts/commit/c837b0bacdc80b6754aaed2f40110e5dadec2a5b))
* extract turn processing helpers from execution.py (Phase 5) ([f87c71a](https://github.com/jpsweeney97/jp-scripts/commit/f87c71a8c843c75b918ece478f0ead6985fdb678))
* Phase 2 - extract evolution/ and system/ to top-level packages ([67e4b91](https://github.com/jpsweeney97/jp-scripts/commit/67e4b9189a3f14e736aa77d917de726bded79195))
* reduce cyclomatic complexity in 4 high-CC modules ([a265790](https://github.com/jpsweeney97/jp-scripts/commit/a265790f5d6ea40633e4b3860a0d572cdb22f63a))
* split memory/store.py into focused submodules (Phase 4) ([de26c5a](https://github.com/jpsweeney97/jp-scripts/commit/de26c5a29af80955c1b48d1e19c2228c99556b7f))


### Documentation

* fix documentation drift across CLI_REFERENCE, HANDBOOK, README ([c567ddc](https://github.com/jpsweeney97/jp-scripts/commit/c567ddc28a4b16763424d4a5064e8cfc4db8ef0b))
* rewrite ARCHITECTURE.md and harden CI constitution enforcement ([2854ee7](https://github.com/jpsweeney97/jp-scripts/commit/2854ee7d63871fae6ab7c6f6d513086615cd1e18))
* update ARCHITECTURE.md to reflect package refactors ([ad0334e](https://github.com/jpsweeney97/jp-scripts/commit/ad0334e1c01c76c9a3d1bac15dd009a2bcb5e3d2))


### CI/CD

* add release-please for automated releases ([28909a4](https://github.com/jpsweeney97/jp-scripts/commit/28909a4b83a934986b6d476fd0749ca2ca1ea229))
* remove PyPI publishing from release workflow ([87e1045](https://github.com/jpsweeney97/jp-scripts/commit/87e10451d75acd5d0a6f6d71b78a49f307062eeb))

## [Unreleased] - Security Hardening & Reliability

### Security

- **Fix:** TOCTOU vulnerability in filesystem tools via new `validate_and_open()` atomic operator that combines path validation with `O_NOFOLLOW` file opening.
- **Fix:** Blocking I/O in security checks; all path validation now has async variants (`validate_path_safe_async`, `validate_workspace_root_safe_async`).
- **Harden:** Enforce strict symlink depth limits (max 10 hops) to prevent symlink chain abuse.
- **Harden:** Forbid access to system root paths (`/etc`, `/usr`, `/bin`, `/sbin`, `/root`, `/System`, `/Library`).
- **Harden:** Circular symlink detection prevents infinite loops during path resolution.

### Core

- **Refactor:** Replaced brittle regex-based JSON parser with a robust stack-based state machine for reliable agent output extraction.
- **Refactor:** Removed `THINKING_PATTERN` regex; thinking tag extraction now uses deterministic string parsing.
- **Add:** `_extract_from_code_fence()` - extracts JSON from markdown fences without regex.
- **Add:** `_extract_thinking_content()` - handles malformed/broken `<thinking>` tags gracefully.
- **Add:** `_find_last_valid_json()` - greedy fallback that finds the last valid JSON object in chatty LLM output.

### Testing

- **Add:** Comprehensive symlink attack test suite (`tests/security/test_symlink_attacks.py`) with 30 tests covering escape attempts, chained symlinks, system directory protection, and TOCTOU mitigations.
- **Add:** Robust JSON extraction tests (`tests/unit/test_robust_json.py`) with 33 tests covering edge cases like nested braces, code in strings, and broken tags.

---

## [0.8.0] - The Architect Update

### Architecture

- Deleted redundant `src/jpscripts/mcp_server.py` entry point; unified server startup via `jpscripts.mcp.server`.

### Core

- Replaced hardcoded `HARD_CONTEXT_CAP` with dynamic, model-aware limits in `TokenBudgetManager`.
- `DEFAULT_MODEL_CONTEXT_LIMIT` (200K) used as fallback; actual limit passed at initialization.

### Memory

- Added `source_path` tracking to `MemoryEntry` schema for file-based memories.
- New `jp memory vacuum` command prunes stale embeddings when source files are deleted.

### Governance

- Codified strict invariants (No Shell Injection, Async Purity, Type Rigidness, Error Containment) in `AGENTS.md` Section 7.

### Removed

- Deleted legacy `mcp_server.py` wrapper.

## [0.7.0] - The Dynamic Update

### Core Architecture

- **Token Budgeting**: Implemented `TokenBudgetManager` to enforce strict priority-based context allocation (Diagnostic > Diffs > Files).
- **Smart Truncation**: Integrated `smart_read_context` into the budget manager to prevent syntax corruption when truncating files.
- **Dynamic Registry**: `jpscripts.mcp.tools` now auto-discovers tool modules at runtime using `pkgutil`, removing the need for manual registration lists.

### Changed

- `prepare_agent_prompt` now strictly adheres to model context limits, dropping low-priority dependencies before truncating critical error logs.

## [0.5.0] - The God-Mode Update

### Changed

- GitHub PR interactions now run fully async (`gpr`/`_get_prs`) with non-blocking gh subprocesses and safer TTY handling for SSH commands.
- Context ingestion keeps JSON valid via structural truncation and YAML-aware dispatch in `smart_read_context`.
- Swarm orchestration uses a `SwarmController` state machine with agent-nominated `next_step`, bounded turns, and structured handoffs.
- Memory scoring now prefers recent entries via time-decayed keyword overlap.
- Process discovery/killing now runs through async, non-blocking psutil wrappers across commands and MCP tools.
- Agents fetch semantic memory even without command output, boosting architecture/security tags from prompts.
- Context gathering warns when commands reference paths outside the workspace; search fallbacks render match panels when fzf is missing.
- Handbook now prefers `jp fix` over legacy aliases and documents piping `jp map` into `jp fix` for refactors.

## [0.4.9] - Unreleased

### Added

- Integration safeguard for Codex handoff with a mocked subprocess and XML prompt validation (`tests/integration/test_agent_real.py`).
- Async git core helpers for remote URL lookup, stash management, and porcelain-short status parsing to replace GitPython-only surfaces.
- `make lint` entry for strict `mypy` gating ahead of tests.
- `jp fix` supports `--loop`/`--max-retries` self-healing runs with automated patch application and optional revert on failure.
- MCP git tool `get_workspace_status` reports branch state across all workspace repositories for external agents.
- Agent system prompts now render through Jinja2 (`src/jpscripts/templates/agent_system.xml.j2`) with a reusable cdata filter.
- MCP filesystem tool `apply_patch` applies unified diffs with workspace validation, pure-Python hunks, and `git apply` fallback.
- Semantic memory embeddings now use a singleton client that prefers a local embedding server (`embedding_server_url`) before loading SentenceTransformer weights.

### Changed

- Git plumbing now uses true asyncio subprocess calls and porcelain v2 parsing for status, fetch, and commit workflows (no GitPython threads).
- `git-extra` commands (`gstage`, `gbrowse`, `stashview`) now run through `AsyncRepo` with asyncio orchestration, removing direct GitPython dependencies.
- Python context reads fall back to warn-tagged head/tail slices when AST parsing fails, preserving context through syntax errors.
- Memory storage now uses a VectorStore adapter with LanceDB + NoOp implementations, simplifying optional dependency handling while retaining keyword fallbacks.
- Agent prompts now auto-inline AGENTS.md as a constitution and require explicit <thinking> blocks in responses.
- Repair loop now detects repeated failures, injects a step-back strategy override, and requests higher reasoning effort/temperature before proposing new patches.
- `jp standup` now uses AsyncRepo-powered async git queries (no GitPython dependency) while honoring author and date filters.
- System commands now inject `AppConfig` instances directly (no global config), and process/port kill flows thread config through Typer contexts and MCP tools.
- Documentation split: README focuses on install/config/CLI reference with handbook link, and root ignores clean up stray Python scripts.

### Removed

- Deleted stray `fake_*.py` root scripts and ignore new root-level Python files by default.

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
