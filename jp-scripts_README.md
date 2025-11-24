# jp-scripts

Personal command toolbox for macOS (Homebrew assumed) – small, composable scripts that remove micro-frictions in daily development.

All scripts live in `~/Projects/jp-scripts/bin` and are on `$PATH`:

```zsh
export PATH="$HOME/Projects/jp-scripts/bin:$PATH"
```

Run `jpcom` anytime to see the current command catalog.

---

## Quick start

1. PATH + deps:
   ```zsh
   export PATH="$HOME/Projects/jp-scripts/bin:$PATH"
   jpdoctor         # general deps
   jpdoctor tools   # deps from registry requires
   ```
2. Discover: `jpcom` or `jpcom fzf`; deep dive with `jpcom info <name>`.
3. Health: `jp-lint` (or `jp-check`), `jp-smoke`.
4. Top commands to try: `proj`, `hist`, `ripper`, `note`, `gprepush`, `gwhatpush`.

---

## Meta & maintenance

| Command            | What                                                                                                  | Requires             |
| ------------------ | ----------------------------------------------------------------------------------------------------- | -------------------- |
| jpcom              | Browse command catalog; `jpcom info <name>` shows tags/requires/examples; `jpcom fzf` is the palette  | bash, jq, fzf        |
| jpdoctor           | Tooling health; `jpdoctor tools` aggregates registry `requires`; `jpdoctor command <name>` checks one | bash, jq             |
| jp-lint / jp-check | Lint registry + headers (+ optional shellcheck); jp-check = jp-lint + jpdoctor                        | bash, jq, shellcheck |
| jp-smoke           | Non-interactive smoke test (lint, doctor, jpcom, note)                                                | bash, jq             |
| jp-sync            | Keep jp-scripts healthy (git status, lint/doctor/smoke, optional snapshot; env: JP_SYNC_REPO/JP_SYNC_SNAP_DIR) | bash, git, jq, zip   |
| jp-bootstrap       | Bootstrap PATH + notes/work dirs + health checks                                                      | bash                 |
| jpnew              | Scaffold a new command (bin stub + registry entry; `--edit` opens it)                                 | bash, jq, fzf        |
| work               | Run workspace scripts from `~/.config/jp-work.d`                                                      | bash                 |
| standup            | Summarize recent git commits across repos (env: STANDUP_ROOT, STANDUP_MAX_DEPTH, STANDUP_EXCLUDES)   | bash, git            |
| envrun             | Run a command with env vars loaded from a file                                                        | bash                 |
| hist               | Fuzzy-search zsh history (most recent first; de-duplicated)                                           | bash, fzf            |

### Optional deps, env overrides, quiet/test knobs

- Optional deps: `fd`/`bat`/`git` speed up `recent`, `todo-scan`, `loggrep`; `gh` is optional for `gbrowse` (falls back to URL open).
- Env overrides (common): `JP_SYNC_REPO`/`JP_SYNC_SNAP_DIR` (jp-sync), `STANDUP_ROOT`/`STANDUP_MAX_DEPTH`/`STANDUP_EXCLUDES` (standup), `JP_NOTES_DIR` (note/note-search/standup-note), `JP_WORKTREE_ROOT` (gworktree), `JP_EDITOR` (note).
- Non-interactive/testing envs: `RECENT_NONINTERACTIVE`, `TODO_SCAN_NONINTERACTIVE`, `LOGGREP_NONINTERACTIVE`, `GWORKTREE_NONINTERACTIVE`, `LOGGREP_FOLLOW_ONCE=1` (one-line follow), `JP_SMOKE_SYNC_PUSH=1` (enable push/pull check in jp-smoke).
- Quiet/dry-run: cleanzip/snaprepo/jp-sync/snap support `--quiet`; `cleanzip`/`snaprepo`/`jp-sync snap` log excludes and human-readable size in normal mode; `log_run` respects `JP_DRY_RUN=1`.
- Smoke controls: `JP_SMOKE=1` (set by jp-smoke) skips redundant jp-sync checks; `JP_SMOKE_MAX_TIME` optionally aborts long runs; `JP_SMOKE_SYNC_PUSH=1` opt-in to push/pull smoke.
- Helper conventions: scripts prefer `lib/log.sh` for info/warn/error, `lib/deps.sh` for `deps_require`/`deps_warn_missing`; fzf helpers set consistent headers/previews.

### Workspaces

Workspace scripts live in `~/.config/jp-work.d/<name>.sh`. Example:

```bash
# ~/.config/jp-work.d/umaf.sh
# UMAF main dev workspace
cd "$HOME/Projects/UMAF" || exit 1
code .
```

Then:

```zsh
work          # list all workspaces
work umaf     # run UMAF workspace script
```

---

## Navigation & search

| Command | What                                                                                     | Requires           |
| ------- | ---------------------------------------------------------------------------------------- | ------------------ |
| proj    | Fuzzy-jump to frequently used directories via zoxide + fzf (use `cd "$(proj)"` or Alt-P) | bash, zoxide, fzf  |
| recent  | Fuzzy-jump to recently modified files/dirs in the current project (fzf + preview)        | bash, fzf, fd, bat |
| ripper  | Interactive code search: rg → fzf with bat preview, jump to editor                       | bash, rg, fzf, bat |
| todo-scan | Scan for TODO/FIXME/HACK/BUG and jump into files (supports git-root, filters)          | bash, rg, fzf, bat |
| loggrep | Friendly log search/tail with rg + fzf preview                                           | bash, rg, fzf, bat |
| hist    | Fuzzy zsh history (see Meta table)                                                       | bash, fzf          |

---

## Git helpers

| Command         | What                                                            | Requires               |
| --------------- | --------------------------------------------------------------- | ---------------------- |
| gwhatpush       | Show what `git push` would do (ahead/behind, commits, diffstat) | bash, git              |
| gprepush        | Pre-push checklist (status, whatpush, optional tests)           | bash, git              |
| gbranches       | List branches with upstream/ahead/behind/last commit            | bash, git              |
| gstatus-all     | Summarize git status across repos under a root                  | bash, git              |
| gclean-branches | List/delete local branches merged into a base                   | bash, git              |
| gundo-last      | Safely undo last commit with upstream safety checks             | bash, git              |
| gpatch          | Apply a unified diff from clipboard/file with a safety check    | bash, git, pbpaste     |
| gstage          | Interactively stage changed files by number                     | bash, git              |
| git-branchcheck | Interactive branch picker (local + remote)                      | bash, git, fzf         |
| stashview       | Interactive stash browser with apply/pop/drop                   | bash, git, fzf         |
| gbrowse         | Open current repo/branch/PR/commit/compare on GitHub            | bash, git (gh optional)|
| snaprepo        | Snapshot current git repo as a clean zip (uses cleanzip)        | bash, git, cleanzip    |
| gworktree       | Manage git worktrees with safe defaults and fzf picker          | bash, git, fzf         |
| gpr             | GitHub PR helper with gh + fzf (view/open/checkout)             | bash, git, gh, fzf, jq |

---

## Files & text

| Command        | What                                                                  | Requires        |
| -------------- | --------------------------------------------------------------------- | --------------- |
| clean-ds       | Remove .DS_Store and AppleDouble files safely (dry-run by default)    | bash, find      |
| cleanzip       | Clean, deterministic-ish zip with sensible excludes                   | bash, zip, find |
| rename-edit    | Bulk rename files/dirs by editing a list in your editor               | bash            |
| global-replace | Safe project-wide search/replace with rg + sd (dry-run default)       | bash, rg, sd    |
| diffview       | Pretty diff between two files or stdin/file using git diff --no-index | bash, git       |

---

## Notes & clipboard

| Command  | What                                                                      | Requires                        |
| -------- | ------------------------------------------------------------------------- | ------------------------------- |
| note     | Fast daily note capture/append in `$HOME/Notes/quick-notes/YYYY-MM-DD.md` | bash                            |
| note-search | Fzf search across daily Markdown notes and jump to a match             | bash, rg, fzf, bat              |
| standup-note | Run standup and append the output into today’s note as a section      | bash, standup, git              |
| cliphist | Clipboard history: save, browse (fzf+bat), restore                        | bash, pbcopy, pbpaste, fzf, bat |

---

## System & processes

| Command       | What                                                 | Requires        |
| ------------- | ---------------------------------------------------- | --------------- |
| process-kill  | Interactive process picker/killer (ps + fzf)         | bash, ps, fzf   |
| port-kill     | Find and kill processes listening on a TCP port      | bash, lsof      |
| brew-explorer | Explore Homebrew formulas/casks with fzf + brew info | bash, brew, fzf |
| dots          | Wrapper around stow for managing dotfiles            | bash, stow      |

---

## macOS / network / web

| Command   | What                                                                | Requires                     |
| --------- | ------------------------------------------------------------------- | ---------------------------- |
| audioswap | Interactive audio output switcher (SwitchAudioSource + fzf)         | bash, SwitchAudioSource, fzf |
| ssh-open  | Fuzzy-launch SSH connections from `~/.ssh/config`                   | bash, ssh, fzf               |
| tmpserver | Simple HTTP server for current directory (`python3 -m http.server`) | bash, python3                |

---

## Registry & metadata

- Command catalog: `registry/commands.json` (name, path, category, description, tags, requires, examples).
- `jpcom info <name>` surfaces description + tags/requires/examples; `jpdoctor tools` aggregates `requires`.
- Keep `bin/<script>` headers in sync with registry; `jp-lint` will warn on missing metadata or drift.
- `jpnew` scaffolds a new script and appends a registry entry (tags/examples included by default).

## Shared helpers & config

- Config (`lib/config.sh`): reads `~/.jpconfig` once; env overrides win (`JP_EDITOR`, `JP_NOTES_DIR`, `JP_FOCUS_AUDIO_DEVICE`, `JP_WORKTREE_ROOT`, `JP_BREW_PROFILE`). Example:

  ```bash
  # ~/.jpconfig
  editor="code -w"
  notes_dir="$HOME/Notes/quick-notes"
  focus_audio_device="Headphones"
  worktree_root="$HOME/Projects/.worktrees"
  brew_profile="work"
  ```

- Logging (`lib/log.sh`): `log_info/warn/error/debug`, `log_run`; honors `JP_VERBOSE`, `JP_DRY_RUN`, and `NO_COLOR/JP_NO_COLOR`.
- Deps (`lib/deps.sh`): `deps_require` / `deps_warn_missing` with Homebrew hints.
- Filesystem (`lib/fs.sh`): `fs_list_by_mtime <root> <include_dirs?> <max?> [excludes...]` uses `fd` when available, falls back to `find` + stat (mtime + size).
- fzf (`lib/fzf.sh`): require helper + common opts, plus `fzf_join_header`, `fzf_default_preview_cmd`, `fzf_add_default_preview`.
- git (`lib/git.sh`): repo checks + root/branch helpers, upstream/ahead/behind, remote URL parsing, worktree introspection.

---

## Zsh keybindings

Optional bindings in `~/.zshrc`:

| Key    | Action                      | Command   |
| ------ | --------------------------- | --------- |
| Ctrl-R | Fuzzy history               | hist      |
| Alt-P  | Project jump                | proj      |
| Alt-O  | Open current dir in VS Code | code .    |
| Alt-J  | Command palette             | jpcom fzf |

Add the widget definitions from this README to your `~/.zshrc` to enable.

---

## Installation / setup

1. Clone or copy `jp-scripts` to `~/Projects/jp-scripts`.
2. Add to PATH in `~/.zshrc`:

   ```zsh
   export PATH="$HOME/Projects/jp-scripts/bin:$PATH"
   ```

3. (Optional) install recommended tools via Homebrew. `jpdoctor` can print a `brew install` line:

   ```zsh
   jpdoctor
   ```

4. Source your shell config:

   ```zsh
   source ~/.zshrc
   ```

5. Explore:

   ```zsh
   jpcom
   ```

   to see everything available.
