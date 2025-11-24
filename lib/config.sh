#!/usr/bin/env bash
# shellcheck shell=bash

# Config loader for jp-scripts.
# Precedence: environment override > ~/.jpconfig value > caller-provided default.
# Known keys: editor, notes_dir, focus_audio_device, worktree_root, brew_profile.
# Format:
#   key=value
#   # comments and blank lines are ignored
#   values may be quoted to allow spaces

JP_CONFIG_PATH_DEFAULT="${HOME}/.jpconfig"

jp_config_path() {
  printf '%s\n' "${JP_CONFIG_FILE:-$JP_CONFIG_PATH_DEFAULT}"
}

jp_config_load() {
  # Populate JP_CONFIG_<key> variables from the config file once.
  if [ "${JP_CONFIG_LOADED:-0}" -eq 1 ]; then
    return 0
  fi

  JP_CONFIG_LOADED=1

  local file
  file="$(jp_config_path)"

  if [ ! -f "$file" ]; then
    return 0
  fi

  local line key value
  while IFS= read -r line || [ -n "$line" ]; do
    # Trim leading/trailing whitespace.
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"

    [ -z "$line" ] && continue
    case "$line" in
      \#*) continue ;;
    esac

    key="${line%%=*}"
    value="${line#*=}"

    # Trim whitespace around key/value.
    key="${key%"${key##*[![:space:]]}"}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"

    # Strip matching quotes to allow spaces.
    if [ "${value#\"}" != "$value" ] && [ "${value%\"}" != "$value" ]; then
      value="${value#\"}"
      value="${value%\"}"
    elif [ "${value%\'}" != "$value" ] && [ "${value#\'}" != "$value" ]; then
      value="${value#\'}"
      value="${value%\'}"
    fi

    case "$key" in
      editor|notes_dir|focus_audio_device|worktree_root|brew_profile)
        local var_name="JP_CONFIG_${key}"
        printf -v "$var_name" '%s' "$value"
        ;;
      *)
        # Unknown keys are ignored for forward compatibility.
        ;;
    esac
  done <"$file"
}

jp_config_get() {
  # Usage: jp_config_get <key> [default] [env_var_override]
  local key="$1"
  local default="${2-}"
  local env_var="${3-}"

  if [ -n "$env_var" ] && [ -n "${!env_var-}" ]; then
    printf '%s\n' "${!env_var}"
    return 0
  fi

  jp_config_load

  local var_name="JP_CONFIG_${key}"
  if [ -n "${!var_name-}" ]; then
    printf '%s\n' "${!var_name}"
    return 0
  fi

  if [ -n "$default" ]; then
    printf '%s\n' "$default"
  fi
}
