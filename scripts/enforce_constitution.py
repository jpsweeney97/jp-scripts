#!/usr/bin/env python3
"""Enforce AGENTS.md constitutional rules on the codebase.

This script walks all Python files in src/ and checks them for constitutional
violations using the governance module's AST-based compliance checker.

Usage:
    python scripts/enforce_constitution.py           # Report mode (default)
    python scripts/enforce_constitution.py --strict  # Fail on violations
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_path))

from jpscripts.core.governance import (
    Violation,
    check_source_compliance,
    format_violations_for_agent,
    has_fatal_violations,
)


def main() -> int:
    """Walk src/ and check all Python files for constitutional violations.

    Returns:
        0 if no fatal violations (or report mode), 1 if strict mode with violations.
    """
    parser = argparse.ArgumentParser(description="Check codebase for constitutional violations")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if fatal violations found (default: report only)",
    )
    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parent.parent / "src"
    all_violations: list[Violation] = []

    print(f"Checking {src_dir} for constitutional violations...")

    file_count = 0
    for py_file in src_dir.rglob("*.py"):
        file_count += 1
        try:
            source = py_file.read_text(encoding="utf-8")
            violations = check_source_compliance(source, py_file)
            all_violations.extend(violations)
        except Exception as e:
            print(f"Error checking {py_file}: {e}", file=sys.stderr)

    print(f"Checked {file_count} files.")

    if all_violations:
        print(format_violations_for_agent(all_violations), file=sys.stderr)
        error_count = sum(1 for v in all_violations if v.severity == "error")
        warning_count = sum(1 for v in all_violations if v.severity == "warning")
        print(f"\nTotal: {error_count} error(s), {warning_count} warning(s)", file=sys.stderr)

        if args.strict and has_fatal_violations(all_violations):
            print("\n❌ Strict mode: failing due to fatal violations", file=sys.stderr)
            return 1

        if not args.strict:
            print("\n⚠️  Report mode: violations logged but not failing CI", file=sys.stderr)
            print("   Run with --strict to enforce violations", file=sys.stderr)
    else:
        print("✓ No constitutional violations detected")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
