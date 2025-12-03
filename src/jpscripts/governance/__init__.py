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

Rules are loaded from safety_rules.yaml in the templates directory.
See governance/config.py for the configuration schema.
"""

from jpscripts.governance.ast_checker import ConstitutionChecker
from jpscripts.governance.compliance import (
    check_compliance,
    check_compliance_async,
    check_source_compliance,
    check_source_compliance_async,
    count_violations_by_severity,
    format_violations_for_agent,
    has_fatal_violations,
    scan_codebase_compliance,
    scan_codebase_compliance_async,
)
from jpscripts.governance.config import SafetyConfig, load_safety_config
from jpscripts.governance.diff_parser import apply_patch_in_memory
from jpscripts.governance.secret_scanner import check_for_secrets
from jpscripts.governance.types import Violation, ViolationType

__all__ = [
    "ConstitutionChecker",
    "SafetyConfig",
    "Violation",
    "ViolationType",
    "apply_patch_in_memory",
    "check_compliance",
    "check_compliance_async",
    "check_for_secrets",
    "check_source_compliance",
    "check_source_compliance_async",
    "count_violations_by_severity",
    "format_violations_for_agent",
    "has_fatal_violations",
    "load_safety_config",
    "scan_codebase_compliance",
    "scan_codebase_compliance_async",
]
