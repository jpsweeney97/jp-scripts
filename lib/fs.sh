#!/usr/bin/env bash
# shellcheck shell=bash

# Filesystem helpers shared across jp-scripts.

FS_STAT_STYLE="gnu"
if stat -f '%m' . >/dev/null 2>&1; then
  FS_STAT_STYLE="bsd"
fi

FS_FD_BIN=""
if command -v fd >/dev/null 2>&1; then
  FS_FD_BIN="fd"
elif command -v fdfind >/dev/null 2>&1; then
  FS_FD_BIN="fdfind"
fi

fs_stat_line() {
  # Usage: fs_stat_line <path>
  # Prints: <mtime_epoch>\t<size_bytes>\t<path>
  local path="$1"
  local meta mtime size

  if [ "$FS_STAT_STYLE" = "bsd" ]; then
    meta="$(stat -f '%m %z' "$path" 2>/dev/null)" || return 1
  else
    meta="$(stat -c '%Y %s' "$path" 2>/dev/null)" || return 1
  fi

  mtime="${meta%% *}"
  size="${meta#* }"

  printf '%s\t%s\t%s\n' "$mtime" "$size" "$path"
}

fs_list_by_mtime() {
  # Usage: fs_list_by_mtime <root> <include_dirs 0|1> <max 0=all> [excludes...]
  # Output: lines of "<mtime>\t<size>\t<path>" sorted by mtime desc.
  local root="$1"
  local include_dirs="${2:-0}"
  local max="${3:-0}"
  shift 3 || true
  local excludes=("$@")

  if [ ! -d "$root" ]; then
    printf 'Error: fs_list_by_mtime root not found: %s\n' "$root" >&2
    return 1
  fi

  root="$(cd "$root" && pwd)"

  gather_cmd() {
    if [ -n "$FS_FD_BIN" ]; then
      local fd_args=(--hidden --follow --exclude .git --absolute-path)
      if [ "$include_dirs" -eq 1 ]; then
        fd_args+=(--type f --type d)
      else
        fd_args+=(--type f)
      fi
      local ex
      for ex in "${excludes[@]}"; do
        fd_args+=(--exclude "$ex")
      done
      "$FS_FD_BIN" "${fd_args[@]}" . "$root"
    else
      local find_prune=("(" "-path" "*/.git" "-o" "-path" "*/.git/*")
      local ex
      for ex in "${excludes[@]}"; do
        find_prune+=("-o" "-path" "*/$ex" "-o" "-path" "*/$ex/*")
      done
      find_prune+=(")")

      local find_types=("(" "-type" "f")
      if [ "$include_dirs" -eq 1 ]; then
        find_types+=("-o" "-type" "d")
      fi
      find_types+=(")")

      find "$root" "${find_prune[@]}" -prune -o "${find_types[@]}" -print 2>/dev/null
    fi
  }

  if [ "$max" -gt 0 ]; then
    gather_cmd \
      | while IFS= read -r path; do
          [ "$path" = "$root" ] && continue
          fs_stat_line "$path"
        done \
      | sort -rn -k1,1 \
      | head -n "$max" || true
  else
    gather_cmd \
      | while IFS= read -r path; do
          [ "$path" = "$root" ] && continue
          fs_stat_line "$path"
        done \
      | sort -rn -k1,1 || true
  fi
}
