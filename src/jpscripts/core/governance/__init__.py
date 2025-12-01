"""
Constitutional compliance checker for jp-scripts.

This package enforces AGENTS.md rules programmatically by parsing diffs
and detecting violations via AST analysis. It implements a "warn + prompt"
strategy where violations are fed back to the agent for correction.

Key invariants enforced:
- No blocking I/O in async context (subprocess.run without asyncio.to_thread)
- No bare except clauses
- No shell=True in subprocess calls
- No untyped Any without type: ignore comment
"""

from jpscripts.core.governance.ast_checker import ConstitutionChecker
from jpscripts.core.governance.compliance import (
    check_compliance,
    check_source_compliance,
    count_violations_by_severity,
    format_violations_for_agent,
    has_fatal_violations,
    scan_codebase_compliance,
)
from jpscripts.core.governance.diff_parser import apply_patch_in_memory
from jpscripts.core.governance.secret_scanner import check_for_secrets
from jpscripts.core.governance.types import Violation, ViolationType

__all__ = [
    "ConstitutionChecker",
    "Violation",
    "ViolationType",
    "apply_patch_in_memory",
    "check_compliance",
    "check_for_secrets",
    "check_source_compliance",
    "count_violations_by_severity",
    "format_violations_for_agent",
    "has_fatal_violations",
    "scan_codebase_compliance",
]
