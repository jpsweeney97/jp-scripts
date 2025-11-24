#!/usr/bin/env bash
# shellcheck shell=bash

# Dependency checks with friendly Homebrew hints.

_deps_error() {
  if command -v log_error >/dev/null 2>&1; then
    log_error "$@"
  else
    printf 'Error: %s\n' "$*" >&2
  fi
}

_deps_warn() {
  if command -v log_warn >/dev/null 2>&1; then
    log_warn "$@"
  else
    printf 'Warning: %s\n' "$*" >&2
  fi
}

deps_require() {
  # Usage: deps_require <cmd> [brew_formula] [friendly_name]
  local cmd="$1"
  local formula="${2-}"
  local friendly="${3-}"

  if command -v "$cmd" >/dev/null 2>&1; then
    return 0
  fi

  local msg="requires $cmd"
  [ -n "$friendly" ] && msg="$msg ($friendly)"
  [ -n "$formula" ] && msg="$msg; install with: brew install $formula"

  _deps_error "$msg"
  return 1
}

deps_warn_missing() {
  # Usage: deps_warn_missing <cmd> [brew_formula] [friendly_name]
  local cmd="$1"
  local formula="${2-}"
  local friendly="${3-}"

  if command -v "$cmd" >/dev/null 2>&1; then
    return 0
  fi

  local msg="$cmd not found"
  [ -n "$friendly" ] && msg="$msg ($friendly)"
  [ -n "$formula" ] && msg="$msg; install with: brew install $formula"

  _deps_warn "$msg"
  return 0
}
