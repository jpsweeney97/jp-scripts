#!/usr/bin/env python3
"""Enforce AGENTS.md constitutional rules on the codebase.

Usage:
    python scripts/enforce_constitution.py           # Report mode (default)
    python scripts/enforce_constitution.py --strict  # Fail on violations
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jpscripts.core.governance import (
    format_violations_for_agent,
    has_fatal_violations,
    scan_codebase_compliance,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check codebase for constitutional violations")
    parser.add_argument("--strict", action="store_true", help="Exit with code 1 on fatal violations")
    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parent.parent / "src"
    print(f"Checking {src_dir} for constitutional violations...")

    violations, file_count = scan_codebase_compliance(src_dir)
    print(f"Checked {file_count} files.")

    if violations:
        print(format_violations_for_agent(violations), file=sys.stderr)
        error_count = sum(1 for v in violations if v.severity == "error")
        warning_count = sum(1 for v in violations if v.severity == "warning")
        print(f"\nTotal: {error_count} error(s), {warning_count} warning(s)", file=sys.stderr)

        if args.strict and has_fatal_violations(violations):
            print("\n❌ Strict mode: failing due to fatal violations", file=sys.stderr)
            return 1
        if not args.strict:
            print("\n⚠️  Report mode: violations logged but not failing CI", file=sys.stderr)
    else:
        print("✓ No constitutional violations detected")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
