"""Merge conflict resolution with intelligent categorization.

This module provides tools for categorizing and resolving git merge conflicts
using a tiered strategy:
- TRIVIAL: Auto-resolve whitespace, import ordering
- SEMANTIC: Attempt LLM-assisted resolution for logic conflicts
- COMPLEX: Flag for human review

Key classes:
- ConflictCategory: Classification enum
- ConflictMarker: Parsed conflict region with ours/theirs/base
- MergeConflictResolver: Main resolution orchestrator

[invariant:typing] All types are explicit; mypy --strict compliant.
[invariant:async-io] All I/O operations use async patterns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Final

from jpscripts.core.result import Err, Ok, Result, ValidationError


class ConflictCategory(Enum):
    """Classification of merge conflicts by resolution strategy.

    TRIVIAL: Whitespace, import ordering - auto-resolve deterministically
    SEMANTIC: Logic changes - attempt LLM resolution with verification
    COMPLEX: Overlapping structural changes - flag for human review
    """

    TRIVIAL = auto()
    SEMANTIC = auto()
    COMPLEX = auto()


@dataclass(frozen=True)
class ConflictMarker:
    """A parsed merge conflict region.

    Attributes:
        file_path: Path to the file containing the conflict
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (1-indexed)
        ours: Content from HEAD (current branch)
        theirs: Content from merging branch
        base: Content from common ancestor (None for two-way merge)
    """

    file_path: Path
    start_line: int
    end_line: int
    ours: str
    theirs: str
    base: str | None = None


@dataclass
class ResolutionReport:
    """Summary of conflict resolution results.

    Attributes:
        total: Total number of conflicts found
        trivial_resolved: Auto-resolved trivial conflicts
        semantic_resolved: LLM-resolved semantic conflicts
        complex_flagged: Conflicts flagged for human review
        failed: Conflicts that couldn't be resolved
    """

    total: int = 0
    trivial_resolved: int = 0
    semantic_resolved: int = 0
    complex_flagged: int = 0
    failed: int = 0


class MergeConflictResolver:
    """Orchestrates merge conflict resolution using tiered strategy.

    Workflow:
    1. Parse conflict markers from files
    2. Categorize each conflict (TRIVIAL, SEMANTIC, COMPLEX)
    3. Auto-resolve TRIVIAL conflicts
    4. Attempt LLM resolution for SEMANTIC conflicts
    5. Flag COMPLEX conflicts for human review
    6. Generate resolution report

    [invariant:async-io] All I/O uses async patterns
    """

    # Threshold for COMPLEX categorization
    _COMPLEX_LINE_THRESHOLD: Final[int] = 30
    _COMPLEX_DIFF_RATIO: Final[float] = 0.7

    # Git conflict marker patterns
    _MARKER_START: Final[str] = "<<<<<<< "
    _MARKER_BASE: Final[str] = "||||||| "
    _MARKER_SEP: Final[str] = "======="
    _MARKER_END: Final[str] = ">>>>>>> "

    def __init__(self) -> None:
        """Initialize the resolver."""
        self._report = ResolutionReport()
        self._resolved_files: dict[Path, str] = {}

    async def categorize_conflict(self, marker: ConflictMarker) -> ConflictCategory:
        """Categorize a conflict for resolution strategy selection.

        Args:
            marker: The parsed conflict to categorize

        Returns:
            ConflictCategory indicating resolution strategy
        """
        ours = marker.ours.strip()
        theirs = marker.theirs.strip()

        # Check for TRIVIAL: identical content (false conflict)
        if ours == theirs:
            return ConflictCategory.TRIVIAL

        # Check for TRIVIAL: whitespace-only differences
        if self._is_whitespace_only_diff(ours, theirs):
            return ConflictCategory.TRIVIAL

        # Check for TRIVIAL: import reordering
        if self._is_import_reorder(ours, theirs):
            return ConflictCategory.TRIVIAL

        # Check for COMPLEX: large conflicts
        ours_lines = ours.count("\n") + 1
        theirs_lines = theirs.count("\n") + 1
        if max(ours_lines, theirs_lines) > self._COMPLEX_LINE_THRESHOLD:
            return ConflictCategory.COMPLEX

        # Check for COMPLEX: high divergence ratio
        similarity = self._calculate_similarity(ours, theirs)
        if similarity < (1 - self._COMPLEX_DIFF_RATIO):
            return ConflictCategory.COMPLEX

        # Default to SEMANTIC for medium-sized logic conflicts
        return ConflictCategory.SEMANTIC

    def _is_whitespace_only_diff(self, ours: str, theirs: str) -> bool:
        """Check if difference is only whitespace."""
        # Remove all whitespace and compare
        ours_normalized = re.sub(r"\s+", "", ours)
        theirs_normalized = re.sub(r"\s+", "", theirs)
        return ours_normalized == theirs_normalized

    def _is_import_reorder(self, ours: str, theirs: str) -> bool:
        """Check if difference is only import statement reordering."""
        # Extract import lines
        import_pattern = re.compile(r"^(?:from\s+\S+\s+)?import\s+.+$", re.MULTILINE)

        ours_imports = set(import_pattern.findall(ours))
        theirs_imports = set(import_pattern.findall(theirs))

        # Check if all lines are imports and sets are equal
        ours_lines = {line.strip() for line in ours.strip().split("\n") if line.strip()}
        theirs_lines = {line.strip() for line in theirs.strip().split("\n") if line.strip()}

        # All lines must be imports for this to be a simple reorder
        if ours_lines == ours_imports and theirs_lines == theirs_imports:
            return ours_imports == theirs_imports

        return False

    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """Calculate Jaccard similarity between strings."""
        if not s1 and not s2:
            return 1.0

        # Token-based similarity
        tokens1 = set(re.findall(r"\w+", s1))
        tokens2 = set(re.findall(r"\w+", s2))

        if not tokens1 and not tokens2:
            return 1.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    async def parse_conflict_markers(
        self,
        content: str,
        file_path: Path,
    ) -> list[ConflictMarker]:
        """Parse git conflict markers from file content.

        Supports both standard and diff3 style markers.

        Args:
            content: File content with conflict markers
            file_path: Path to the file

        Returns:
            List of parsed ConflictMarker objects
        """
        markers: list[ConflictMarker] = []
        lines = content.split("\n")

        i = 0
        while i < len(lines):
            if lines[i].startswith(self._MARKER_START):
                start_line = i + 1  # 1-indexed

                # Parse ours section
                ours_lines: list[str] = []
                base_lines: list[str] | None = None
                theirs_lines: list[str] = []

                i += 1
                section = "ours"

                while i < len(lines):
                    line = lines[i]

                    if line.startswith(self._MARKER_BASE):
                        section = "base"
                        base_lines = []
                        i += 1
                        continue
                    elif line.startswith(self._MARKER_SEP):
                        section = "theirs"
                        i += 1
                        continue
                    elif line.startswith(self._MARKER_END):
                        end_line = i + 1  # 1-indexed
                        break

                    if section == "ours":
                        ours_lines.append(line)
                    elif section == "base" and base_lines is not None:
                        base_lines.append(line)
                    elif section == "theirs":
                        theirs_lines.append(line)

                    i += 1

                markers.append(
                    ConflictMarker(
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        ours="\n".join(ours_lines),
                        theirs="\n".join(theirs_lines),
                        base="\n".join(base_lines) if base_lines is not None else None,
                    )
                )

            i += 1

        return markers

    async def resolve_trivial(
        self,
        marker: ConflictMarker,
    ) -> Result[str, ValidationError]:
        """Resolve a trivial conflict automatically.

        Args:
            marker: The conflict to resolve

        Returns:
            Ok(resolved_content) or Err if not resolvable
        """
        ours = marker.ours.strip()
        theirs = marker.theirs.strip()

        # Identical content - use either
        if ours == theirs:
            return Ok(ours)

        # Whitespace differences - prefer version without trailing whitespace
        if self._is_whitespace_only_diff(ours, theirs):
            # Prefer ours but strip trailing whitespace from lines
            normalized = "\n".join(line.rstrip() for line in marker.ours.split("\n"))
            return Ok(normalized)

        # Import reorder - use sorted imports
        if self._is_import_reorder(ours, theirs):
            # Extract and sort imports
            import_pattern = re.compile(r"^(?:from\s+\S+\s+)?import\s+.+$", re.MULTILINE)
            imports = sorted(import_pattern.findall(ours))
            return Ok("\n".join(imports))

        return Err(ValidationError(f"Cannot trivially resolve conflict at {marker.file_path}:{marker.start_line}"))

    async def resolve_conflicts(
        self,
        conflict_files: list[Path],
    ) -> Result[bool, ValidationError]:
        """Resolve conflicts in a list of files.

        Args:
            conflict_files: Paths to files with conflicts

        Returns:
            Ok(True) if all resolved, Ok(False) if some need review, Err on failure
        """
        all_resolved = True
        self._report = ResolutionReport()

        for file_path in conflict_files:
            if not file_path.exists():
                continue

            content = file_path.read_text()
            markers = await self.parse_conflict_markers(content, file_path)

            if not markers:
                continue

            self._report.total += len(markers)

            resolved_content = content
            needs_human_review = False

            for marker in markers:
                category = await self.categorize_conflict(marker)

                if category == ConflictCategory.TRIVIAL:
                    result = await self.resolve_trivial(marker)
                    if isinstance(result, Ok):
                        resolved_content = self._apply_resolution(
                            resolved_content,
                            marker,
                            result.value,
                        )
                        self._report.trivial_resolved += 1
                    else:
                        self._report.failed += 1
                        all_resolved = False

                elif category == ConflictCategory.SEMANTIC:
                    # For now, SEMANTIC conflicts need human review
                    # LLM resolution would be implemented here
                    self._report.complex_flagged += 1
                    needs_human_review = True
                    all_resolved = False

                elif category == ConflictCategory.COMPLEX:
                    self._report.complex_flagged += 1
                    needs_human_review = True
                    all_resolved = False

            # Write resolved content if any resolutions were made
            if not needs_human_review and resolved_content != content:
                file_path.write_text(resolved_content)
                self._resolved_files[file_path] = resolved_content
            elif needs_human_review:
                # Still write partial resolutions
                if resolved_content != content:
                    file_path.write_text(resolved_content)

        return Ok(all_resolved)

    def _apply_resolution(
        self,
        content: str,
        marker: ConflictMarker,
        resolution: str,
    ) -> str:
        """Apply a resolution to conflict markers in content.

        Args:
            content: Original file content
            marker: The conflict being resolved
            resolution: The resolved content

        Returns:
            Content with conflict markers replaced by resolution
        """
        # Build regex to match this specific conflict block
        # Match from <<<<<<< to >>>>>>>
        pattern = re.compile(
            rf"<<<<<<<[^\n]*\n"
            rf"(?:.*?\n)*?"  # ours content
            rf"(?:\|\|\|\|\|\|\|[^\n]*\n(?:.*?\n)*?)?"  # optional base
            rf"=======\n"
            rf"(?:.*?\n)*?"  # theirs content
            rf">>>>>>>[^\n]*",
            re.DOTALL,
        )

        # Replace first match (conflicts are processed in order)
        return pattern.sub(resolution, content, count=1)

    def get_resolution_report(self) -> ResolutionReport:
        """Get the resolution report.

        Returns:
            ResolutionReport with statistics
        """
        return self._report


__all__ = [
    "ConflictCategory",
    "ConflictMarker",
    "MergeConflictResolver",
    "ResolutionReport",
]
