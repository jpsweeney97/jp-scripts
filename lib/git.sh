#!/usr/bin/env bash
# shellcheck shell=bash

# Lightweight git helpers for jp-scripts. Source this file from scripts that
# need common git checks and metadata.

git_require() {
  if ! command -v git >/dev/null 2>&1; then
    printf 'Error: git not found on PATH.\n' >&2
    return 1
  fi
}

git_is_repo() {
  git rev-parse --is-inside-work-tree >/dev/null 2>&1
}

git_require_repo() {
  git_require || return 1
  if ! git_is_repo; then
    printf 'Error: must be run inside a git repository.\n' >&2
    return 1
  fi
}

git_repo_root() {
  git rev-parse --show-toplevel 2>/dev/null
}

git_current_branch() {
  git rev-parse --abbrev-ref HEAD 2>/dev/null || printf 'HEAD'
}

git_upstream_ref() {
  local ref="${1:-HEAD}"
  git rev-parse --abbrev-ref --symbolic-full-name "${ref}@{upstream}" 2>/dev/null || true
}

git_ahead_behind() {
  local upstream="$1"
  local ref="${2:-HEAD}"

  if [ -z "$upstream" ]; then
    printf '0 0\n'
    return 0
  fi

  git rev-list --left-right --count "${upstream}...${ref}" 2>/dev/null || printf '0 0\n'
}

git_short_status() {
  git status -sb 2>/dev/null || true
}
