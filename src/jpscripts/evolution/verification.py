"""Test collection and verification for evolution.

This module provides:
- collect_dependent_tests: Find tests affected by changed files
- run_verification: Execute verification tests
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from jpscripts.analysis.structure import get_import_dependencies
from jpscripts.core.config import AppConfig
from jpscripts.core.console import get_logger
from jpscripts.core.result import Err, Ok, Result
from jpscripts.system import run_safe_shell

from .types import EvolutionError, VerificationResult

logger = get_logger(__name__)


async def collect_dependent_tests(
    python_changes: list[Path],
    tests_root: Path,
    root: Path,
) -> list[Path]:
    """Collect test files that depend on the changed Python files.

    Args:
        python_changes: List of changed Python file paths
        tests_root: Root directory for tests
        root: Workspace root directory

    Returns:
        List of test file paths that depend on the changed files
    """
    if not tests_root.exists():
        return []
    try:
        test_files = await asyncio.to_thread(lambda: list(tests_root.rglob("test_*.py")))
    except OSError:
        return []

    dependents: list[Path] = []
    for test_file in test_files:
        try:
            deps: set[Path] = await asyncio.to_thread(get_import_dependencies, test_file, root)
        except Exception as exc:
            logger.debug("Failed to get dependencies for %s: %s", test_file, exc)
            deps = set()
        for changed in python_changes:
            if changed in deps:
                dependents.append(test_file)
                break
    return dependents


async def run_verification(
    changed_paths: list[str],
    root: Path,
    config: AppConfig,
) -> Result[VerificationResult, EvolutionError]:
    """Run verification tests on the evolution branch.

    Determines changed files, collects dependent tests, and runs pytest.

    Args:
        changed_paths: List of changed file paths from git diff
        root: Workspace root directory
        config: Application configuration

    Returns:
        Ok(VerificationResult) on success or if tests pass/fail normally
        Err(EvolutionError) if verification cannot be performed
    """
    # Determine Python changes
    python_changes = [Path(root / p).resolve() for p in changed_paths if p.endswith(".py")]
    tests_root = root / "tests"

    # Collect test targets
    test_targets: list[Path] = []
    test_targets.extend([p for p in python_changes if tests_root in p.parents])
    dependent_tests = await collect_dependent_tests(python_changes, tests_root, root)
    test_targets.extend(dependent_tests)
    if not test_targets and tests_root.exists():
        test_targets.append(tests_root)
    # Preserve order while deduplicating
    test_targets = list(dict.fromkeys(test_targets))

    # Fail fast if pytest is unavailable
    pytest_cmd = "pytest"
    if await asyncio.to_thread(shutil.which, "pytest") is None:
        return Err(EvolutionError("pytest is not available; aborting evolution."))

    test_args = (
        " ".join(str(path.relative_to(root)) for path in test_targets) if test_targets else ""
    )
    test_command = f"{pytest_cmd} -q {test_args}".strip()

    test_result = await run_safe_shell(test_command, root, "evolve.verify", config=config)
    if isinstance(test_result, Err):
        return Err(EvolutionError(f"Test execution failed: {test_result.error}"))

    result_payload = test_result.value
    return Ok(
        VerificationResult(
            test_command=test_command,
            exit_code=result_payload.returncode,
            success=result_payload.returncode == 0,
            output=result_payload.stdout or result_payload.stderr,
        )
    )


__all__ = ["collect_dependent_tests", "run_verification"]
