#!/usr/bin/env bash
# shellcheck shell=bash

# Shared fzf helpers for jp-scripts. Provides a consistent set of defaults and
# simple presence checks without mutating shell options.

fzf_require() {
  if ! command -v fzf >/dev/null 2>&1; then
    printf 'Error: this command requires fzf. Install with: brew install fzf\n' >&2
    return 1
  fi
}

fzf_set_common_opts() {
  # Populate FZF_COMMON_OPTS array with consistent defaults.
  local prompt="${1:-fzf> }"
  local header="${2:-}"
  local preview_window="${3:-}"

  FZF_COMMON_OPTS=(--ansi --height="${FZF_HEIGHT:-80%}" --reverse --prompt="$prompt")
  [ -n "$header" ] && FZF_COMMON_OPTS+=(--header="$header")
  [ -n "$preview_window" ] && FZF_COMMON_OPTS+=(--preview-window="$preview_window")
}

fzf_add_preview() {
  # Usage: fzf_add_preview "<preview command>" [window]
  local preview_cmd="$1"
  local window="${2:-right,60%,border-left}"

  FZF_COMMON_OPTS+=(--preview="$preview_cmd" --preview-window="$window")
}

fzf_join_header() {
  # Build a multi-line fzf header string from individual lines.
  local header=""
  local line
  for line in "$@"; do
    [ -z "$line" ] && continue
    if [ -z "$header" ]; then
      header="$line"
    else
      header="${header}\n${line}"
    fi
  done

  printf '%s' "$header"
}

fzf_default_preview_cmd() {
  # Returns a preview command that favors bat and falls back to sed.
  local max_lines="${1:-200}"

  if command -v bat >/dev/null 2>&1; then
    printf 'bat --style=numbers --color=always --line-range :%s {}' "$max_lines"
  else
    printf 'sed -n "1,%sp" {}' "$max_lines"
  fi
}

fzf_add_default_preview() {
  # Usage: fzf_add_default_preview [max_lines] [window]
  local max_lines="${1:-200}"
  local window="${2:-right,60%,border-left}"

  fzf_add_preview "$(fzf_default_preview_cmd "$max_lines")" "$window"
}
