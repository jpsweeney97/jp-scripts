# jpscripts (Python CLI)

A modern Python 3.11+ rewrite of the old `jp-scripts` toolbox. Commands are implemented with Typer, Rich, GitPython, and Pydantic—no legacy Bash wrappers required.

## Install
- Editable: `pip install -e .`
- Isolated (recommended): `pipx install .`

## Quick usage
- Meta: `jp com`, `jp doctor`, `jp config`, `jp version`
- Git: `jp status-all`, `jp whatpush`, `jp gundo-last`, `jp gstage`, `jp gpr`, `jp gbrowse`, `jp git-branchcheck`, `jp stashview`
- Navigation/search: `jp recent`, `jp proj`, `jp ripper`, `jp todo-scan`, `jp loggrep`
- Notes/productivity: `jp note`, `jp note-search`, `jp standup`, `jp standup-note`, `jp cliphist`
- System: `jp process-kill`, `jp port-kill`, `jp brew-explorer`, `jp audioswap`, `jp ssh-open`, `jp tmpserver`
- Init: `jp init` (interactive config bootstrap)

### Old → new mapping
- `jpcom` → `jp com`
- `jpdoctor` → `jp doctor`
- `gstatus-all` → `jp status-all`
- `gwhatpush` → `jp whatpush`
- `recent` → `jp recent`
- `proj` → `jp proj`
- `jp-bootstrap` → `jp init`
- Other legacy git helpers → see Git commands above

## Configuration
Config is read from `~/.jpconfig` (TOML) or the path set by `JPSCRIPTS_CONFIG`; environment variables override individual keys (`JP_EDITOR`, `JP_NOTES_DIR`, `JP_WORKSPACE_ROOT`, `JP_SNAPSHOTS_DIR`, `JP_LOG_LEVEL`, `JP_WORKTREE_ROOT`, `JP_FOCUS_AUDIO_DEVICE`).

Example `~/.jpconfig`:
```toml
editor = "code -w"
notes_dir = "/Users/me/Notes/quick-notes"
workspace_root = "/Users/me/Projects"
worktree_root = "/Users/me/Projects/.worktrees"
snapshots_dir = "/Users/me/snapshots"
log_level = "INFO"
focus_audio_device = "Headphones"
```

Run `jp init` to generate this interactively.

## Development
- Stack: Python 3.11+, Typer, Rich, GitPython, Pydantic, psutil, pyperclip.
- Layout: `src/jpscripts/` for code, `tests/` for pytest-based smoke tests.
- Commands live in `src/jpscripts/commands/` and are registered in `src/jpscripts/main.py`.
- Tests: `pytest` (basic smoke tests in `tests/test_smoke.py`).

## Legacy
The original Bash-based `bin/`, `lib/`, and `registry/` are preserved under `legacy/` for reference only. The Python CLI is the source of truth going forward.
