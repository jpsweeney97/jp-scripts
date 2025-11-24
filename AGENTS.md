# AGENTS.md

## Project Overview

This repository contains a personal command toolbox for macOS (Homebrew assumed). It consists of composable Bash scripts located in `bin/` and shared helpers in `lib/`.

## Directory Structure

- `bin/`: Executable scripts. All scripts here must be on the `$PATH`.
- `lib/`: Shared Bash libraries (`log.sh`, `deps.sh`, `config.sh`, `fzf.sh`, `git.sh`).
- `registry/`: Contains `commands.json`, the source of truth for command metadata.

## Development Standards

### 1. Script Format

All new scripts in `bin/` must adhere to the following standards:

- **Shebang & Safety**: Must start with `#!/usr/bin/env bash` and `set -euo pipefail`.
- **Headers**: Must include a standard comment header block for documentation parsing:

  ```bash
  # Name: <script-name>
  # Category: <category>
  # Description: <short summary>
  # Usage:
  #   <script-name> [flags]
  ```

- **Helpers**: Source shared libraries using the standard repo-root resolution pattern found in existing scripts:

  ```bash
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)"
  REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
  . "$REPO_ROOT/lib/log.sh"
  . "$REPO_ROOT/lib/deps.sh"
  ```

### 2. Registry Registration

Every executable in `bin/` **must** have a corresponding entry in `registry/commands.json`.

- When creating a new script, you must append a JSON object to the `commands` array.
- Required fields: `name`, `path` (`bin/<name>`), `category`, `description`, `tags` (array), `requires` (array of binaries), `examples` (array).

### 3. Dependencies

- Do not assume tools are installed. Use `deps_require` from `lib/deps.sh` to enforce dependencies.
- Example: `deps_require jq jq` (checks for `jq`, suggests `brew install jq`).

### 4. Output & Logging

- Use `lib/log.sh` for all status output.
- Use `log_info`, `log_warn`, `log_error`, and `log_fatal`.
- Avoid raw `echo` for logging unless printing data for a pipe.

## Verification & Testing

Before declaring a task complete, run the following health checks:

1.  **Linting**: Run `bin/jp-lint`.

    - This checks `shellcheck` (if installed), verifies script headers, and ensures `registry/commands.json` matches the contents of `bin/`.
    - Fix any errors regarding missing metadata or registry drift.

2.  **Smoke Tests**: Run `bin/jp-smoke`.

    - This runs a non-interactive suite of tests to verify the runtime health of core utilities.

## Common Tasks

- **Creating a new script**:

  1.  Create `bin/<name>`.
  2.  `chmod +x bin/<name>`.
  3.  Add entry to `registry/commands.json`.
  4.  Run `bin/jp-lint` to verify.

- **Refactoring**:

  - If you move logic into `lib/`, ensure you update the `requires` array in `commands.json` if the dependency requirements change.
