"""
Cyclomatic complexity analysis using McCabe algorithm.

This module provides AST-based complexity analysis for Python files,
extending the structure.py patterns. It computes McCabe cyclomatic
complexity and integrates with the memory system to calculate
technical debt scores.

McCabe Cyclomatic Complexity:
- Base complexity: 1
- Each decision point adds 1: if, for, while, except, with, and/or, comprehension
"""

from __future__ import annotations

import ast
import asyncio
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from jpscripts.core.config import AppConfig
from jpscripts.core.console import get_logger
from jpscripts.core.git import AsyncRepo
from jpscripts.core.memory import query_memory
from jpscripts.core.result import Err, JPScriptsError, Ok, Result

logger = get_logger(__name__)


class ComplexityError(JPScriptsError):
    """Raised when complexity analysis fails."""

    pass


@dataclass(frozen=True)
class FunctionComplexity:
    """Complexity metrics for a single function or method."""

    name: str
    lineno: int
    end_lineno: int
    cyclomatic: int
    is_async: bool = False


@dataclass(frozen=True)
class FileComplexity:
    """Aggregated complexity metrics for a file."""

    path: Path
    functions: tuple[FunctionComplexity, ...]
    total_cyclomatic: int
    max_cyclomatic: int
    average_cyclomatic: float


@dataclass(frozen=True)
class TechnicalDebtScore:
    """Technical debt score for a file.

    Debt Score = Complexity Score x (1 + Fix Frequency) x log(1 + Git Churn)

    Higher scores indicate files that are complex, frequently changed,
    and often fixed, making them prime candidates for refactoring.
    """

    path: Path
    complexity_score: float  # Normalized complexity (max function complexity)
    fix_frequency: int  # Number of times this file appears in fix-related memories
    churn: int  # Number of commits touching the file
    debt_score: float  # complexity_score * (1 + fix_frequency) * log(1 + churn)
    reasons: tuple[str, ...]  # Human-readable reasons for the score


class McCabeVisitor(ast.NodeVisitor):
    """AST visitor implementing McCabe cyclomatic complexity.

    McCabe complexity is computed as:
    - Start with 1 (base complexity)
    - Add 1 for each decision point:
        - if, elif
        - for, while
        - except handler
        - with statement
        - and/or boolean operators
        - comprehension clauses (for, if)
        - ternary expressions (... if ... else ...)
        - assert statements
    """

    def __init__(self) -> None:
        self.complexity: int = 1  # Base complexity

    def visit_If(self, node: ast.If) -> None:
        """Each if/elif adds 1."""
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        """Each for loop adds 1."""
        self.complexity += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        """Each async for loop adds 1."""
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        """Each while loop adds 1."""
        self.complexity += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """Each except handler adds 1."""
        self.complexity += 1
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        """Each with statement adds 1."""
        self.complexity += 1
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        """Each async with statement adds 1."""
        self.complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        """Each and/or adds complexity based on number of values.

        `a and b and c` has 2 decision points (between 3 values).
        """
        # n values means n-1 operators
        self.complexity += len(node.values) - 1
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        """Each comprehension clause adds 1, plus 1 per if filter."""
        self.complexity += 1  # for the for clause
        self.complexity += len(node.ifs)  # for each if filter
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        """Ternary expression (x if cond else y) adds 1."""
        self.complexity += 1
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        """Assert statement adds 1 (implicit branch)."""
        self.complexity += 1
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        """Match statement: each case adds 1."""
        self.complexity += len(node.cases)
        self.generic_visit(node)


def _compute_function_complexity(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Compute McCabe complexity for a single function."""
    visitor = McCabeVisitor()
    visitor.visit(node)
    return visitor.complexity


def analyze_file_complexity_sync(path: Path) -> Result[FileComplexity, ComplexityError]:
    """Analyze cyclomatic complexity of all functions in a Python file.

    Args:
        path: Path to the Python file

    Returns:
        FileComplexity with metrics for all functions
    """
    if not path.exists():
        return Err(ComplexityError(f"File not found: {path}"))

    if path.suffix.lower() != ".py":
        return Err(ComplexityError(f"Not a Python file: {path}"))

    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        return Err(ComplexityError(f"Failed to read file: {exc}"))

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return Err(ComplexityError(f"Syntax error in file: {exc}"))

    functions: list[FunctionComplexity] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            complexity = _compute_function_complexity(node)
            end_lineno = node.end_lineno if node.end_lineno else node.lineno
            functions.append(
                FunctionComplexity(
                    name=node.name,
                    lineno=node.lineno,
                    end_lineno=end_lineno,
                    cyclomatic=complexity,
                    is_async=isinstance(node, ast.AsyncFunctionDef),
                )
            )

    if not functions:
        return Ok(
            FileComplexity(
                path=path,
                functions=(),
                total_cyclomatic=0,
                max_cyclomatic=0,
                average_cyclomatic=0.0,
            )
        )

    total = sum(f.cyclomatic for f in functions)
    max_cc = max(f.cyclomatic for f in functions)
    avg = total / len(functions)

    return Ok(
        FileComplexity(
            path=path,
            functions=tuple(sorted(functions, key=lambda f: -f.cyclomatic)),
            total_cyclomatic=total,
            max_cyclomatic=max_cc,
            average_cyclomatic=avg,
        )
    )


async def analyze_file_complexity(path: Path) -> Result[FileComplexity, ComplexityError]:
    """Async wrapper for file complexity analysis."""
    return await asyncio.to_thread(analyze_file_complexity_sync, path)


async def analyze_directory_complexity(
    root: Path,
    ignore_dirs: Sequence[str],
) -> Result[list[FileComplexity], ComplexityError]:
    """Analyze complexity for all Python files in a directory.

    Args:
        root: Root directory to analyze
        ignore_dirs: Directory names to skip

    Returns:
        List of FileComplexity for all analyzed files
    """
    root = root.expanduser().resolve()
    if not root.exists():
        return Err(ComplexityError(f"Directory not found: {root}"))

    python_files: list[Path] = []
    ignore_set = set(ignore_dirs)

    for path in root.rglob("*.py"):
        # Check if any parent directory should be ignored
        parts = path.relative_to(root).parts
        if any(part in ignore_set for part in parts[:-1]):
            continue
        python_files.append(path)

    results: list[FileComplexity] = []
    for path in python_files:
        match await analyze_file_complexity(path):
            case Err(err):
                logger.debug("Skipping %s due to error: %s", path, err)
                continue
            case Ok(complexity):
                results.append(complexity)

    return Ok(results)


def _query_fix_frequency(path: Path, config: AppConfig) -> int:
    """Query memory for fix-related entries mentioning this file.

    Returns the number of memory entries related to fixes for this file.
    """
    try:
        # Query for the file name/path in fix-related context
        query = f"fix error bug {path.name}"
        memories = query_memory(query, limit=20, config=config)

        # Count how many mention this specific file
        path_str = str(path)
        file_name = path.name
        count = 0
        for memory in memories:
            if path_str in memory or file_name in memory:
                count += 1

        return count
    except Exception as exc:
        logger.debug("Fix frequency query failed for %s: %s", path, exc)
        return 0


async def calculate_debt_scores(
    root: Path,
    config: AppConfig,
) -> Result[list[TechnicalDebtScore], JPScriptsError]:
    """
    Calculate technical debt scores for all Python files.

    Technical Debt Score = Complexity Score x (1 + Fix Frequency)

    The formula ensures that:
    - Complex files with no fix history still get a base score
    - Files with frequent fixes get multiplied scores
    - Very complex AND frequently fixed files bubble to the top

    Args:
        root: Workspace root to analyze
        config: Application configuration

    Returns:
        List of TechnicalDebtScore sorted by debt_score descending
    """
    repo: AsyncRepo | None = None
    match await AsyncRepo.open(root):
        case Ok(open_repo):
            repo = open_repo
        case Err(repo_err):
            logger.debug("Git repository unavailable for churn calculation: %s", repo_err)

    match await analyze_directory_complexity(root, config.ignore_dirs):
        case Err(complexity_err):
            return Err(complexity_err)
        case Ok(complexities):
            pass

    if not complexities:
        return Ok([])

    scores: list[TechnicalDebtScore] = []

    for file_complexity in complexities:
        if file_complexity.max_cyclomatic == 0:
            continue

        # Query fix frequency from memory
        fix_frequency = await asyncio.to_thread(_query_fix_frequency, file_complexity.path, config)

        churn = 0
        if repo is not None:
            match await repo.get_file_churn(file_complexity.path):
                case Ok(value):
                    churn = value
                case Err(err):
                    logger.debug("Failed to calculate churn for %s: %s", file_complexity.path, err)

        # Compute debt score
        complexity_score = float(file_complexity.max_cyclomatic)
        debt_score = complexity_score * (1 + fix_frequency) * math.log(1 + churn)

        # Generate reasons
        reasons: list[str] = []
        if file_complexity.max_cyclomatic > 10:
            most_complex = file_complexity.functions[0]
            reasons.append(
                f"High complexity function: {most_complex.name} (CC={most_complex.cyclomatic})"
            )
        if fix_frequency > 0:
            reasons.append(f"Fix frequency: {fix_frequency} related memory entries")
        if churn > 0:
            reasons.append(f"Churn: {churn} commits touch this file")
        if file_complexity.average_cyclomatic > 5:
            reasons.append(f"High average complexity: {file_complexity.average_cyclomatic:.1f}")

        scores.append(
            TechnicalDebtScore(
                path=file_complexity.path,
                complexity_score=complexity_score,
                fix_frequency=fix_frequency,
                churn=churn,
                debt_score=debt_score,
                reasons=tuple(reasons),
            )
        )

    # Sort by debt score descending
    scores.sort(key=lambda s: -s.debt_score)

    return Ok(scores)


def format_complexity_report(complexities: Sequence[FileComplexity]) -> str:
    """Format complexity analysis as a readable report."""
    if not complexities:
        return "No Python files analyzed."

    lines = ["## Complexity Report", ""]

    # Top 10 most complex files
    sorted_by_max = sorted(complexities, key=lambda c: -c.max_cyclomatic)[:10]

    lines.append("### Top 10 Most Complex Files")
    lines.append("")
    for fc in sorted_by_max:
        if fc.max_cyclomatic == 0:
            continue
        lines.append(
            f"- **{fc.path.name}** (max CC={fc.max_cyclomatic}, "
            f"avg={fc.average_cyclomatic:.1f}, total={fc.total_cyclomatic})"
        )
        for func in fc.functions[:3]:  # Top 3 functions
            lines.append(f"  - `{func.name}`: CC={func.cyclomatic}")

    return "\n".join(lines)


__all__ = [
    "ComplexityError",
    "FileComplexity",
    "FunctionComplexity",
    "McCabeVisitor",
    "TechnicalDebtScore",
    "analyze_directory_complexity",
    "analyze_file_complexity",
    "analyze_file_complexity_sync",
    "calculate_debt_scores",
    "format_complexity_report",
]
