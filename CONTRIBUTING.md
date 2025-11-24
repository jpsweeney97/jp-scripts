# Contributing to jpscripts

## Architecture

This is a **Typer** application organized by domain:

- `src/jpscripts/main.py`: Registry of all commands.
- `src/jpscripts/commands/`: Implementation modules (e.g., `git_ops.py`, `nav.py`).
- `src/jpscripts/core/`: Shared logic (config, git wrappers, console).

## How to add a new command

1.  **Create the logic:** Add a new function in the appropriate `src/jpscripts/commands/` module.
    - _Must_ accept `ctx: typer.Context`.
    - _Must_ use `console.print()` for output.
2.  **Register it:** Import the function in `src/jpscripts/main.py` and register it:
    ```python
    app.command("my-new-cmd")(module.my_new_cmd)
    ```
3.  **Document it:** Add the command to `README.md` immediately.
4.  **Test it:** Add a simple invocation test in `tests/test_smoke.py`.

## Development

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```
