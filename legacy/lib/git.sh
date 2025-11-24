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

git_default_remote() {
  git_is_repo || return 1

  if git remote get-url origin >/dev/null 2>&1; then
    printf 'origin\n'
    return 0
  fi

  git remote 2>/dev/null | head -n 1 || true
}

git_remote_url() {
  local remote="${1:-}"
  git_is_repo || return 1

  if [ -z "$remote" ]; then
    remote="$(git_default_remote)"
  fi

  git config --get "remote.${remote}.url" 2>/dev/null || true
}

git_remote_https_url() {
  local remote="${1:-}"
  local url host path

  git_is_repo || return 1

  url="$(git_remote_url "$remote")"
  [ -n "$url" ] || return 1

  case "$url" in
    git@*:* )
      host="${url#git@}"
      host="${host%%:*}"
      path="${url#*:}"
      ;;
    ssh://git@* )
      path="${url#ssh://git@}"
      host="${path%%/*}"
      path="${path#*/}"
      ;;
    http://*|https://*)
      url="${url%.git}"
      printf '%s\n' "$url"
      return 0
      ;;
    *)
      printf '%s\n' "$url"
      return 0
      ;;
  esac

  path="${path%.git}"
  printf 'https://%s/%s\n' "$host" "$path"
}

git_branch_exists() {
  local branch="$1"
  git_is_repo || return 1
  git show-ref --verify --quiet "refs/heads/$branch"
}

git_is_detached() {
  git_is_repo || return 1

  local ref
  ref="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  [ "$ref" = "HEAD" ]
}

git_is_dirty() {
  git_is_repo || return 1
  git status --porcelain 2>/dev/null | grep -q .
}

git_worktree_list() {
  git_is_repo || return 1

  git worktree list --porcelain 2>/dev/null \
    | awk '
        $1 == "worktree" { path=$0; sub("^worktree ", "", path) }
        $1 == "branch" {
          branch=$2
          sub("^refs/heads/","",branch)
          printf "%s\t%s\n", path, branch
          path=""
          branch=""
        }
        $1 == "detached" && path != "" { printf "%s\t%s\n", path, "detached" ; path=""; branch="" }
      '
}

git_worktree_path_for_branch() {
  local branch="$1"
  git_is_repo || return 1

  git_worktree_list | awk -F '\t' -v b="$branch" '$2 == b { print $1; exit }'
}

git_worktree_is_dirty() {
  local path="$1"
  git -C "$path" rev-parse --is-inside-work-tree >/dev/null 2>&1 || return 1
  git -C "$path" status --porcelain 2>/dev/null | grep -q .
}
