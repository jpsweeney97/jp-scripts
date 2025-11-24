---
description: Scaffold a new Typer command in jpscripts
argument-hint: NAME=<command-name> DOMAIN=<system|nav|git> DESCRIPTION=<help-text>
---

You are a Principal Software Engineer. Execute the following "God-Mode" refactor to add a new command:

1.  **Create Logic**: Create a function `$NAME` in `src/jpscripts/commands/$DOMAIN.py`.
    - Signature: `(ctx: typer.Context, ...)`
    - Docstring: "$DESCRIPTION"
2.  **Register**: Import and add `app.command("$NAME")(module.$NAME)` in `src/jpscripts/main.py`.
3.  **Test**: Create a test case in `tests/unit/test_$DOMAIN.py`.
4.  **Verify**: Run `pytest tests/unit/test_$DOMAIN.py` to ensure stability.

User intent: Create command `$NAME` in domain `$DOMAIN` with description "$DESCRIPTION".
