# Changelog

All notable changes in this work-in-progress branch.

## Unreleased

### Added

- New shared helpers: `lib/git.sh` (git repo checks, branch/upstream/ahead/behind, status) and `lib/fzf.sh` (fzf presence + common option setup).
- New scaffolder: `bin/jpnew` creates a standard command stub, updates registry, and can open the file (`--edit`).
- Registry metadata expansion: added tags/requires/examples across all commands in `registry/commands.json`.
- Documentation: refactored `jp-scripts_README.md` for quicker onboarding (5-minute setup, top commands), task-oriented tables, metadata primer, and condensed keybindings.
- New changelog (`CHANGELOG.md`).

### Changed

- `bin/jpcom`: `info` now surfaces tags/requires/examples; uses fzf helper for the palette.
- `bin/jpdoctor`: added `tools` (aggregate requires status) and `command <name>` modes; keeps the existing static checklist (brew-formula oriented) alongside the registry-driven modes.
- Multiple scripts now source helpers:
  - Git: `gbranches`, `gwhatpush`, `gstatus-all`, `gprepush`, `gclean-branches`, `gundo-last`, `gstage`, `gpatch`, `snaprepo`, `standup`, `git-branchcheck`, `stashview`, `gpr` (git checks + upstream/status helpers).
  - fzf: `hist`, `proj`, `cliphist`, `stashview`, `git-branchcheck`, `gpr`, `process-kill`, `audioswap`, `ssh-open`, `brew-explorer`, `ripper`, `jpcom`, `jpnew` (consistent fzf checks/options).
- README tables compacted (What/Requires combined) and keybindings note tightened.

### Fixed

- Consistent dependency checks and upstream/ahead/behind logic across git helpers.
- Avoided duplicated rg checks in `ripper`; standardized fzf option usage across interactive tools.
