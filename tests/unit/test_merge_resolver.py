"""Tests for merge conflict resolution with categorization."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest

from jpscripts.core.merge_resolver import (
    ConflictCategory,
    ConflictMarker,
    MergeConflictResolver,
)
from jpscripts.core.result import Err, Ok


class TestConflictCategory:
    """Test ConflictCategory enum values."""

    def test_trivial_category_exists(self) -> None:
        """TRIVIAL category for whitespace/import conflicts."""
        assert ConflictCategory.TRIVIAL is not None

    def test_semantic_category_exists(self) -> None:
        """SEMANTIC category for logic conflicts requiring LLM."""
        assert ConflictCategory.SEMANTIC is not None

    def test_complex_category_exists(self) -> None:
        """COMPLEX category for human review."""
        assert ConflictCategory.COMPLEX is not None


class TestConflictMarker:
    """Test ConflictMarker model."""

    def test_create_conflict_marker(self) -> None:
        """Create a ConflictMarker with required fields."""
        marker = ConflictMarker(
            file_path=Path("src/foo.py"),
            start_line=10,
            end_line=20,
            ours="def foo(): pass",
            theirs="def foo(): return 1",
            base="def foo(): pass",
        )
        assert marker.file_path == Path("src/foo.py")
        assert marker.start_line == 10
        assert marker.end_line == 20
        assert marker.ours == "def foo(): pass"
        assert marker.theirs == "def foo(): return 1"
        assert marker.base == "def foo(): pass"

    def test_marker_without_base(self) -> None:
        """ConflictMarker can have None base for two-way merges."""
        marker = ConflictMarker(
            file_path=Path("src/bar.py"),
            start_line=5,
            end_line=10,
            ours="import os",
            theirs="import sys",
            base=None,
        )
        assert marker.base is None


class TestConflictCategorization:
    """Test conflict categorization logic."""

    @pytest.mark.asyncio
    async def test_categorize_whitespace_as_trivial(self) -> None:
        """Whitespace-only differences should be TRIVIAL."""
        resolver = MergeConflictResolver()
        marker = ConflictMarker(
            file_path=Path("src/foo.py"),
            start_line=1,
            end_line=3,
            ours="def foo():\n    pass\n",
            theirs="def foo():\n    pass  \n",  # Trailing whitespace
            base="def foo():\n    pass\n",
        )
        category = await resolver.categorize_conflict(marker)
        assert category == ConflictCategory.TRIVIAL

    @pytest.mark.asyncio
    async def test_categorize_import_reorder_as_trivial(self) -> None:
        """Import reordering should be TRIVIAL."""
        resolver = MergeConflictResolver()
        marker = ConflictMarker(
            file_path=Path("src/foo.py"),
            start_line=1,
            end_line=3,
            ours="import os\nimport sys\n",
            theirs="import sys\nimport os\n",
            base="import os\n",
        )
        category = await resolver.categorize_conflict(marker)
        assert category == ConflictCategory.TRIVIAL

    @pytest.mark.asyncio
    async def test_categorize_logic_change_as_semantic(self) -> None:
        """Logic changes should be SEMANTIC for LLM resolution."""
        resolver = MergeConflictResolver()
        marker = ConflictMarker(
            file_path=Path("src/foo.py"),
            start_line=1,
            end_line=5,
            ours="def calc(x):\n    return x + 1\n",
            theirs="def calc(x):\n    return x * 2\n",
            base="def calc(x):\n    return x\n",
        )
        category = await resolver.categorize_conflict(marker)
        assert category == ConflictCategory.SEMANTIC

    @pytest.mark.asyncio
    async def test_categorize_large_conflict_as_complex(self) -> None:
        """Large conflicts with many changes should be COMPLEX."""
        resolver = MergeConflictResolver()
        # Large conflict with significant structural differences
        ours = "\n".join([f"line {i} ours" for i in range(50)])
        theirs = "\n".join([f"line {i} theirs" for i in range(50)])
        base = "\n".join([f"line {i} base" for i in range(50)])
        marker = ConflictMarker(
            file_path=Path("src/big.py"),
            start_line=1,
            end_line=100,
            ours=ours,
            theirs=theirs,
            base=base,
        )
        category = await resolver.categorize_conflict(marker)
        assert category == ConflictCategory.COMPLEX


class TestTrivialResolution:
    """Test auto-resolution of trivial conflicts."""

    @pytest.mark.asyncio
    async def test_resolve_whitespace_conflict(self) -> None:
        """Whitespace conflicts should be auto-resolved."""
        resolver = MergeConflictResolver()
        marker = ConflictMarker(
            file_path=Path("src/foo.py"),
            start_line=1,
            end_line=2,
            ours="x = 1  \n",  # Trailing whitespace
            theirs="x = 1\n",
            base="x = 1\n",
        )
        result = await resolver.resolve_trivial(marker)
        assert isinstance(result, Ok)
        # Should resolve to the version without trailing whitespace
        assert result.value.strip() == "x = 1"

    @pytest.mark.asyncio
    async def test_resolve_identical_content(self) -> None:
        """Identical content (false conflict) should resolve easily."""
        resolver = MergeConflictResolver()
        marker = ConflictMarker(
            file_path=Path("src/foo.py"),
            start_line=1,
            end_line=2,
            ours="x = 1\n",
            theirs="x = 1\n",
            base="x = 1\n",
        )
        result = await resolver.resolve_trivial(marker)
        assert isinstance(result, Ok)
        assert "x = 1" in result.value


class TestConflictParsing:
    """Test parsing git conflict markers from files."""

    @pytest.mark.asyncio
    async def test_parse_conflict_markers(self) -> None:
        """Parse standard git conflict markers."""
        resolver = MergeConflictResolver()
        content = """def foo():
<<<<<<< HEAD
    return 1
=======
    return 2
>>>>>>> feature
"""
        markers = await resolver.parse_conflict_markers(content, Path("test.py"))
        assert len(markers) == 1
        assert "return 1" in markers[0].ours
        assert "return 2" in markers[0].theirs

    @pytest.mark.asyncio
    async def test_parse_multiple_conflicts(self) -> None:
        """Parse file with multiple conflict regions."""
        resolver = MergeConflictResolver()
        content = """<<<<<<< HEAD
x = 1
=======
x = 2
>>>>>>> feature
def foo():
    pass
<<<<<<< HEAD
y = 3
=======
y = 4
>>>>>>> feature
"""
        markers = await resolver.parse_conflict_markers(content, Path("test.py"))
        assert len(markers) == 2

    @pytest.mark.asyncio
    async def test_parse_diff3_style_markers(self) -> None:
        """Parse diff3 style markers with base section."""
        resolver = MergeConflictResolver()
        content = """<<<<<<< HEAD
x = 1
||||||| base
x = 0
=======
x = 2
>>>>>>> feature
        """
        markers = await resolver.parse_conflict_markers(content, Path("test.py"))
        assert len(markers) == 1
        assert "x = 1" in markers[0].ours
        base_section = markers[0].base
        assert base_section is None or "x = 0" in base_section
        assert "x = 2" in markers[0].theirs


class TestResolveConflicts:
    """Test full conflict resolution workflow."""

    @pytest.mark.asyncio
    async def test_resolve_trivial_conflicts_auto(self, tmp_path: Path) -> None:
        """Trivial conflicts should be auto-resolved."""
        # Create a file with trivial whitespace conflict
        conflict_file = tmp_path / "whitespace.py"
        conflict_file.write_text("""<<<<<<< HEAD
x = 1
=======
x = 1
>>>>>>> feature
""")
        resolver = MergeConflictResolver()
        result = await resolver.resolve_conflicts([conflict_file])

        assert isinstance(result, Ok)
        # File should be resolved
        resolved_content = conflict_file.read_text()
        assert "<<<<<<" not in resolved_content

    @pytest.mark.asyncio
    async def test_complex_conflicts_flagged(self, tmp_path: Path) -> None:
        """Complex conflicts should be flagged for human review."""
        # Create a file with complex conflict
        conflict_file = tmp_path / "complex.py"
        ours = "\n".join([f"ours_line_{i}" for i in range(30)])
        theirs = "\n".join([f"theirs_line_{i}" for i in range(30)])
        conflict_file.write_text(f"""<<<<<<< HEAD
{ours}
=======
{theirs}
>>>>>>> feature
""")
        resolver = MergeConflictResolver()
        result = await resolver.resolve_conflicts([conflict_file])

        # Should return Err or indicate human review needed
        # Complex conflicts can't be auto-resolved
        if isinstance(result, Ok):
            assert result.value is False  # Indicates manual review needed
        else:
            assert isinstance(result, Err)


class TestResolutionReport:
    """Test resolution reporting."""

    @pytest.mark.asyncio
    async def test_get_resolution_report(self, tmp_path: Path) -> None:
        """Generate a resolution report."""
        conflict_file = tmp_path / "test.py"
        conflict_file.write_text("""<<<<<<< HEAD
x = 1
=======
x = 2
>>>>>>> feature
""")
        resolver = MergeConflictResolver()
        await resolver.resolve_conflicts([conflict_file])

        report = resolver.get_resolution_report()
        assert hasattr(report, "total")
        assert report.total >= 0
