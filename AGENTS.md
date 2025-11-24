# AGENTS.md

## Project Overview

`jpscripts` is a Python 3.11+ command toolbox for macOS. The legacy Bash scripts are kept only for reference under `legacy/`; all active commands live in the Python package.

## Directory Structure

- `src/jpscripts/` – Python package
  - `core/` – shared utilities (console, config, git helpers)
  - `commands/` – Typer subcommands (git, nav, system, notes, search, init)
  - `main.py` – Typer app entrypoint (`jp`)
- `tests/` – pytest-based smoke tests
- `legacy/` – archived `bin/`, `lib/`, and `registry/` from the Bash era

## Development Standards

### Commands

- Use Typer functions in `src/jpscripts/commands/`. Keep implementations small and composable; prefer helpers in `core/` for reuse.
- Register new commands in `src/jpscripts/main.py` via `app.command("name")(function)`.
- Use Rich for user-facing output; avoid plain `print` unless emitting machine-readable data.
- Validate external tools gracefully (e.g., check `shutil.which`) and provide install hints.

### Configuration

- Configuration is modeled with Pydantic in `core/config.py`. Respect defaults and environment overrides. Do not read arbitrary files directly; use `load_config`.

### Dependencies

- Runtime deps are declared in `pyproject.toml` under `[project.dependencies]`; dev/test deps live under `[project.optional-dependencies].dev`.
- Prefer Python libraries (`pathlib`, `psutil`, `asyncio`, `subprocess`) over shelling out where feasible.

## Verification & Testing

- Run `pytest` for smoke coverage (`tests/test_smoke.py`). Add focused tests for new modules where appropriate.
- Keep tests isolated from user state by using the fixtures in `tests/conftest.py` (they stub config and console).

## Adding a Command (quick recipe)

1. Create `src/jpscripts/commands/<name>.py` or add to an existing module.
2. Implement a Typer command function that accepts `typer.Context` when config/logging is needed.
3. Register it in `src/jpscripts/main.py`.
4. Add a short README note if it’s user-facing.
5. Add or extend a pytest to cover the basics.
