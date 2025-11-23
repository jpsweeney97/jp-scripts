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
| jp-bootstrap       | Bootstrap PATH + notes/work dirs + health checks                                                      | bash                 |
| jpnew              | Scaffold a new command (bin stub + registry entry; `--edit` opens it)                                 | bash, jq, fzf        |
| work               | Run workspace scripts from `~/.config/jp-work.d`                                                      | bash                 |
| standup            | Summarize recent git commits across repos                                                             | bash, git            |
| envrun             | Run a command with env vars loaded from a file                                                        | bash                 |
| hist               | Fuzzy-search zsh history (most recent first; de-duplicated)                                           | bash, fzf            |

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
| ripper  | Interactive code search: rg → fzf with bat preview, jump to editor                       | bash, rg, fzf, bat |
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
| snaprepo        | Snapshot current git repo as a clean zip (uses cleanzip)        | bash, git, cleanzip    |
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
