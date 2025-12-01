"""Package entry point for running jpscripts as a module.

Allows the package to be executed via `python -m jpscripts`.
"""

from __future__ import annotations

from .main import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
