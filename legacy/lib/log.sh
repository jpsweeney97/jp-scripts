#!/usr/bin/env bash
# shellcheck shell=bash

# Lightweight logging helpers for jp-scripts.
# Honors:
#   JP_VERBOSE=1   -> enable verbose/debug logs
#   JP_DRY_RUN=1   -> log commands instead of running them when using log_run
#   NO_COLOR/JP_NO_COLOR -> disable ANSI color

LOG_COLOR_ENABLED=1

if [ -n "${NO_COLOR:-}" ] || [ -n "${JP_NO_COLOR:-}" ]; then
  LOG_COLOR_ENABLED=0
elif ! [ -t 1 ] && ! [ -t 2 ]; then
  LOG_COLOR_ENABLED=0
fi

_log_color_wrap() {
  local color="$1"
  local text="$2"

  if [ "$LOG_COLOR_ENABLED" -eq 1 ]; then
    printf '\033[%sm%s\033[0m' "$color" "$text"
  else
    printf '%s' "$text"
  fi
}

_log_print() {
  local level="$1"
  local color="$2"
  local stream="$3"
  shift 3

  local prefix
  prefix="$(_log_color_wrap "$color" "$level")"

  # shellcheck disable=SC2059
  printf '%s: %s\n' "$prefix" "$*" >&"$stream"
}

log_info() {
  _log_print "info" "34" 1 "$@"
}

log_warn() {
  _log_print "warn" "33" 2 "$@"
}

log_error() {
  _log_print "error" "31" 2 "$@"
}

log_debug() {
  if [ "${JP_VERBOSE:-0}" -eq 1 ]; then
    _log_print "debug" "90" 1 "$@"
  fi
}

log_run() {
  # Usage: log_run <command> [args...]
  # Respects JP_DRY_RUN=1 to avoid side effects in scripts.
  local cmd=("$@")

  if [ "${JP_DRY_RUN:-0}" -eq 1 ]; then
    log_info "dry-run: ${cmd[*]}"
    return 0
  fi

  "${cmd[@]}"
}

log_fatal() {
  _log_print "error" "31" 2 "$@"
  exit 1
}
