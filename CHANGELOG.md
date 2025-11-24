# Changelog

All notable changes in this work-in-progress branch.

## Unreleased

### Added

- New shared helpers: `lib/config.sh` (env + ~/.jpconfig loader), `lib/log.sh` (colorized logging, dry-run/verbose hooks), `lib/deps.sh` (require/warn with brew hints), `lib/fs.sh` (mtime-sorted listings with fd/find+stat), and expanded `lib/fzf.sh` helpers (headers + bat/sed preview).
- `lib/git.sh` now covers remote URL parsing + worktree inspection in addition to repo checks/branch/upstream/ahead/behind/status.
- New nav/search scripts: `recent` (recently-touched files picker), `todo-scan` (TODO/FIXME/HACK/BUG navigator with git-aware filters), and `loggrep` (log search/tail with rg + fzf).
- New git utilities: `gbrowse` (open repo/branch/PR/commit/compare on GitHub) and `gworktree` (worktree list/add/fzf/prune helper).
- New notes/meta scripts: `note-search` (fzf over daily notes; can append text and create today’s note on no-match), `standup-note` (append standup output into today’s note), and `jp-sync` (health runner + snapshot helper for jp-scripts; optional push and summary).
- `jp-sync` honors `JP_SYNC_REPO`/`JP_SYNC_SNAP_DIR` for temp runs; `cleanzip`/`snaprepo` gained `--quiet`, size summary, and now use shared log/deps. `jp-sync` help notes relevant env vars; smoke covers snapshots.
- New scaffolder: `bin/jpnew` creates a standard command stub, updates registry, and can open the file (`--edit`).
- Registry metadata expansion: added tags/requires/examples across all commands in `registry/commands.json`.
- Documentation: refactored `jp-scripts_README.md` for quicker onboarding (5-minute setup, top commands), task-oriented tables, metadata primer, and condensed keybindings.
- New changelog (`CHANGELOG.md`).
- Standup adds `STANDUP_MAX_DEPTH`/`STANDUP_EXCLUDES`; registry marks `gh` optional for `gbrowse`; README documents optional deps/env overrides/helper conventions.

### Changed

- `bin/jpcom`: `info` now surfaces tags/requires/examples; uses fzf helper for the palette.
- `bin/jpdoctor`: added `tools` (aggregate requires status) and `command <name>` modes; keeps the existing static checklist (brew-formula oriented) alongside the registry-driven modes.
- Multiple scripts now source helpers:
  - Git: `gbranches`, `gwhatpush`, `gstatus-all`, `gprepush`, `gclean-branches`, `gundo-last`, `gstage`, `gpatch`, `snaprepo`, `standup`, `git-branchcheck`, `stashview`, `gpr` (git checks + upstream/status helpers).
  - fzf: `hist`, `proj`, `cliphist`, `stashview`, `git-branchcheck`, `gpr`, `process-kill`, `audioswap`, `ssh-open`, `brew-explorer`, `ripper`, `jpcom`, `jpnew` (consistent fzf checks/options).
- README tables compacted (What/Requires combined) and keybindings note tightened; added shared helper/config docs + optional deps/env overrides section.
- `jp-smoke` exercises helper libraries and widened coverage (recent/todo-scan/loggrep/gworktree/gbrowse); optional jp-sync push/pull check can be enabled via `JP_SMOKE_SYNC_PUSH=1`.
- Cleanzip/snaprepo/jp-sync summary output standardized (excludes + human-readable size); cleanzip `--quiet` is silent; loggrep follow mode supports `LOGGREP_FOLLOW_ONCE=1` for smokes.
- `ai-pack` hardened: repo-root aware selection with size column/preview, size caps with prompt override, temp-file streaming, pbcopy fallback, registry/README entry, and optional dep hints.

### Fixed

- Consistent dependency checks and upstream/ahead/behind logic across git helpers.
- Avoided duplicated rg checks in `ripper`; standardized fzf option usage across interactive tools.
