# Changelog

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
