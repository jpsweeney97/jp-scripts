# jp-scripts

Personal command toolbox for macOS – small, composable scripts that remove micro-frictions in daily development.

All scripts live in `~/Projects/jp-scripts/bin` and are on `$PATH`:

```zsh
export PATH="$HOME/Projects/jp-scripts/bin:$PATH"
```

Run `jpcom` anytime to see the current command catalog.

---

## Core meta tools

- `jpcom` – list all custom JP commands and what they do.
- `jpdoctor` – check presence/versions of key CLI tools and suggest `brew install` lines.
- `hist` – fuzzy-search full zsh history (most recent first; de-duplicated; optional `--exec`).
- `standup` – summarize recent git commits across repos for daily standup.
- `work` – run workspace setup scripts from `~/.config/jp-work.d` (per-project “spin up”).
- `envrun` – run a command with environment variables loaded from a file (e.g. `.env`).

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

## Files & text

- `clean-ds` – safely remove `.DS_Store` and AppleDouble (`._*`) files under a directory.
- `cleanzip` – create a clean, deterministic-ish zip of a directory, excluding build/cache/junk/secret files.
- `rename-edit` – bulk rename files/directories by editing a list in your editor.
- `global-replace` – safe project-wide search & replace with `rg` + `sd` (dry-run by default).
- `diffview` – pretty diff between two files or stdin/file using `git diff --no-index`.

---

## Git helpers

- `gbranches` – show local branches with upstream, ahead/behind, and last commit info.
- `gclean-branches` – list or delete local branches already merged into a base branch.
- `git-branchcheck` – interactive git branch picker; switches or creates tracking branches.
- `gpatch` – apply a unified diff from clipboard or a file with a safety check.
- `gprepush` – run pre-push checklist (status, whatpush, optional tests).
- `gstage` – interactively stage changed files by number.
- `gstatus-all` – summarize git status for all repos under a root directory.
- `gundo-last` – safely undo the last commit with upstream safety checks and confirmation.
- `gwhatpush` – show what `git push` would do (ahead/behind, commits, diffstat).
- `snaprepo` – snapshot the current git repo as a clean zip using `cleanzip`.
- `stashview` – interactive git stash browser with apply/pop/drop actions.

---

## Navigation & search

- `proj` – fuzzy-jump to frequently used directories via `zoxide` + `fzf`.
  - Use `cd "$(proj)"` or the Alt-P keybinding.
- `ripper` – interactive `rg` + `fzf` code search with `bat` preview and jump-to-editor.

---

## Notes & clipboard

- `note` – fast daily note capture/append in a date-stamped Markdown file
  (`$HOME/Notes/quick-notes/YYYY-MM-DD.md`).
- `cliphist` – clipboard history manager: save, browse, and restore clipboard entries.

---

## System & processes

- `process-kill` – interactive process picker+killer using `ps` + `fzf`.
- `port-kill` – find and kill processes listening on a given TCP port.
- `brew-explorer` – explore Homebrew formulas and casks with `fzf` and `brew info`.
- `dots` – friendly wrapper around `stow` for managing dotfiles (`$HOME/dotfiles`).

---

## macOS / network / web

- `audioswap` – interactive audio output device switcher using `SwitchAudioSource` + `fzf`.
- `ssh-open` – fuzzy-launch SSH connections from `~/.ssh/config` using `fzf`.
- `tmpserver` – start a simple HTTP server for the current directory (`python3 -m http.server`).

---

## Zsh keybindings

These live in `~/.zshrc` and hook into jp-scripts:

```zsh
# Ctrl-R: fuzzy history via hist
fzf-history-widget() {
  local selected
  selected="$(hist)" || return 0
  if [ -n "$selected" ]; then
    LBUFFER="$selected"
    CURSOR=${#LBUFFER}
  fi
}
zle -N fzf-history-widget
bindkey '^R' fzf-history-widget

# Alt-P: project jump via proj (zoxide)
proj-cd-widget() {
  local dir
  dir="$(proj)" || return 0
  [ -z "$dir" ] && return 0
  cd "$dir" || return 0
  zle reset-prompt
}
zle -N proj-cd-widget
bindkey '^[p' proj-cd-widget

# Alt-O: open current directory in VS Code
code-here-widget() {
  if command -v code >/dev/null 2>&1; then
    code .
  else
    echo "code: VS Code CLI not found" >&2
  fi
  zle reset-prompt
}
zle -N code-here-widget
bindkey '^[o' code-here-widget
```

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
