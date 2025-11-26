# JPScripts Architectural Constitution

## Type Safety
- All Python code must pass `mypy --strict`. No `Any` allowed unless documented.

## Import Strategy
- Use `from __future__ import annotations`. Sort imports with `isort` rules.

## Error Handling
- Ban bare `except:`. Always catch specific exceptions.

## Testing
- Every new command in `src/jpscripts/commands/` requires a corresponding unit test in `tests/unit/`.
