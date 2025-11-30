from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from jpscripts.core.context import (
    TRUNCATION_MARKER,
    TokenBudgetManager,
    TokenCounter,
    get_file_skeleton,
    read_file_context,
    smart_read_context,
)


class MockTokenCounter(TokenCounter):
    def __init__(self) -> None:
        super().__init__(default_model="gpt-4o")

    def count_tokens(self, text: str, model: str | None = None) -> int:
        return len(text)

    def tokens_to_characters(self, tokens: int) -> int:
        return tokens

    def trim_to_fit(self, text: str, max_tokens: int, model: str | None = None) -> str:
        return text[:max_tokens]


def test_read_file_context_truncates(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"
    path.write_text("abcd" * 100, encoding="utf-8")

    result = read_file_context(path, max_chars=10)
    assert result == "abcdabcdab"


def test_read_file_context_handles_binary(tmp_path: Path) -> None:
    path = tmp_path / "bin.dat"
    path.write_bytes(b"\xff\x00\xfe")

    result = read_file_context(path, max_chars=10)
    assert result is None


def test_smart_read_context_aligns_to_definition(tmp_path: Path) -> None:
    source = "def first():\n    return 'a'\n\ndef second():\n    return 'b'\n"
    path = tmp_path / "module.py"
    path.write_text(source, encoding="utf-8")

    snippet = smart_read_context(path, max_chars=200)

    assert "def first" in snippet
    assert "def second" in snippet
    ast.parse(snippet)


def test_get_file_skeleton_replaces_long_bodies(tmp_path: Path) -> None:
    source = (
        "def big():\n"
        '    """docstring"""\n'
        "    a = 1\n"
        "    b = 2\n"
        "    c = 3\n"
        "    d = 4\n"
        "    return a + b + c + d\n"
    )
    path = tmp_path / "skeleton.py"
    path.write_text(source, encoding="utf-8")

    skeleton = get_file_skeleton(path)

    assert "def big" in skeleton
    assert "pass" in skeleton or "..." in skeleton
    assert "return a + b + c + d" not in skeleton
    ast.parse(skeleton)


def test_smart_read_context_structured_json(tmp_path: Path) -> None:
    payload = {"a": 1, "b": 2}
    text = json.dumps(payload, indent=2)
    path = tmp_path / "data.json"
    path.write_text(text, encoding="utf-8")

    snippet = smart_read_context(path, max_chars=len(text) - 2)

    assert len(snippet) <= len(text) - 2
    if snippet:
        json.loads(snippet)


# ---------------------------------------------------------------------------
# TokenBudgetManager Tests
# ---------------------------------------------------------------------------


class TestTokenBudgetManagerInit:
    """Tests for TokenBudgetManager initialization and validation."""

    def test_init_with_valid_budgets(self) -> None:
        mgr = TokenBudgetManager(total_budget=100, reserved_budget=10)
        assert mgr.total_budget == 100
        assert mgr.reserved_budget == 10
        assert mgr.remaining() == 90

    def test_init_validates_negative_total(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            TokenBudgetManager(total_budget=-1)

    def test_init_validates_negative_reserved(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            TokenBudgetManager(total_budget=100, reserved_budget=-1)

    def test_init_validates_reserved_exceeds_total(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed"):
            TokenBudgetManager(total_budget=100, reserved_budget=200)

    def test_init_with_zero_budgets(self) -> None:
        mgr = TokenBudgetManager(total_budget=0, reserved_budget=0)
        assert mgr.remaining() == 0


class TestTokenBudgetManagerAllocate:
    """Tests for TokenBudgetManager.allocate method."""

    def test_allocate_within_budget_returns_full_content(self) -> None:
        mgr = TokenBudgetManager(total_budget=100, token_counter=MockTokenCounter())
        result = mgr.allocate(1, "hello")
        assert result == "hello"
        assert mgr.remaining() == 95

    def test_allocate_exceeding_budget_truncates(self) -> None:
        mgr = TokenBudgetManager(total_budget=50, token_counter=MockTokenCounter())
        content = "a" * 100
        result = mgr.allocate(1, content)
        assert len(result) <= 50
        assert TRUNCATION_MARKER in result

    def test_allocate_empty_content_returns_empty(self) -> None:
        mgr = TokenBudgetManager(total_budget=100, token_counter=MockTokenCounter())
        result = mgr.allocate(1, "")
        assert result == ""
        assert mgr.remaining() == 100

    def test_allocate_zero_budget_returns_empty(self) -> None:
        mgr = TokenBudgetManager(total_budget=0, token_counter=MockTokenCounter())
        result = mgr.allocate(1, "content")
        assert result == ""

    def test_allocate_exhausted_budget_returns_empty(self) -> None:
        mgr = TokenBudgetManager(total_budget=10, token_counter=MockTokenCounter())
        mgr.allocate(1, "0123456789")  # Consume all budget
        result = mgr.allocate(2, "more content")
        assert result == ""

    def test_allocate_with_path_uses_smart_truncation(self, tmp_path: Path) -> None:
        """When source_path is provided, smart_read_context is used for truncation."""
        py_file = tmp_path / "sample.py"
        py_file.write_text("def foo():\n    return 1\n", encoding="utf-8")

        mgr = TokenBudgetManager(total_budget=1000, token_counter=MockTokenCounter())
        content = py_file.read_text()
        result = mgr.allocate(1, content, source_path=py_file)
        assert "def foo" in result


class TestTokenBudgetManagerTruncation:
    """Tests for content truncation behavior."""

    def test_truncate_adds_marker(self) -> None:
        mgr = TokenBudgetManager(
            total_budget=25, token_counter=MockTokenCounter()
        )  # Less than content length
        content = "line1\nline2\nline3\nline4\nline5\n"  # 30 chars
        result = mgr.allocate(1, content)
        assert TRUNCATION_MARKER in result

    def test_truncate_prefers_line_boundary(self) -> None:
        mgr = TokenBudgetManager(total_budget=40, token_counter=MockTokenCounter())
        content = "short\n" + "x" * 50
        result = mgr.allocate(1, content)
        # Should truncate at the newline boundary if reasonable
        assert result.startswith("short")

    def test_truncate_too_small_returns_empty(self) -> None:
        mgr = TokenBudgetManager(
            total_budget=5, token_counter=MockTokenCounter()
        )  # Less than marker length
        result = mgr.allocate(1, "some content")
        assert result == ""


class TestTokenBudgetManagerTracking:
    """Tests for budget tracking and summary."""

    def test_remaining_decreases_after_allocation(self) -> None:
        mgr = TokenBudgetManager(total_budget=100, token_counter=MockTokenCounter())
        assert mgr.remaining() == 100
        mgr.allocate(1, "12345")
        assert mgr.remaining() == 95
        mgr.allocate(2, "67890")
        assert mgr.remaining() == 90

    def test_reserved_reduces_available(self) -> None:
        mgr = TokenBudgetManager(
            total_budget=100, reserved_budget=30, token_counter=MockTokenCounter()
        )
        assert mgr.remaining() == 70

    def test_summary_tracks_by_priority(self) -> None:
        mgr = TokenBudgetManager(total_budget=100, token_counter=MockTokenCounter())
        mgr.allocate(1, "first")
        mgr.allocate(3, "third")
        mgr.allocate(2, "second")

        summary = mgr.summary()
        assert summary["priority_1"] == 5
        assert summary["priority_2"] == 6
        assert summary["priority_3"] == 5

    def test_summary_starts_at_zero(self) -> None:
        mgr = TokenBudgetManager(total_budget=100, token_counter=MockTokenCounter())
        summary = mgr.summary()
        assert summary == {"priority_1": 0, "priority_2": 0, "priority_3": 0}


class TestTokenBudgetManagerPriority:
    """Tests for priority-based allocation behavior."""

    def test_priority_allocations_are_independent(self) -> None:
        """Each priority can receive allocations independently."""
        mgr = TokenBudgetManager(total_budget=100, token_counter=MockTokenCounter())

        mgr.allocate(1, "high")
        mgr.allocate(2, "medium")
        mgr.allocate(3, "low")

        summary = mgr.summary()
        assert summary["priority_1"] == 4
        assert summary["priority_2"] == 6
        assert summary["priority_3"] == 3

    def test_budget_shared_across_priorities(self) -> None:
        """All priorities share the same total budget."""
        mgr = TokenBudgetManager(total_budget=20, token_counter=MockTokenCounter())

        mgr.allocate(1, "0123456789")  # 10 chars
        assert mgr.remaining() == 10

        mgr.allocate(2, "abcde")  # 5 chars
        assert mgr.remaining() == 5

        mgr.allocate(3, "XY")  # 2 chars
        assert mgr.remaining() == 3
