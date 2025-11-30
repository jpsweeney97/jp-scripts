"""Property-based tests for TokenBudgetManager using Hypothesis.

These tests verify core invariants of the TokenBudgetManager:
- Allocated tokens never exceed available budget
- Handles arbitrary unicode without crashing
- Empty strings return empty without consuming budget
- Zero budget returns empty strings
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from jpscripts.core.tokens import TokenBudgetManager

if TYPE_CHECKING:
    from jpscripts.core.tokens import Priority

# === Strategies ===

# Budget values (0 to 1M)
budget_strategy = st.integers(min_value=0, max_value=1_000_000)
reserved_strategy = st.integers(min_value=0, max_value=1_000_000)

# Text content (ASCII + Unicode, excluding surrogates)
# Cast to satisfy mypy's strict literal type checking
_SURROGATE_CATEGORIES: tuple[str, ...] = ("Cs",)
text_strategy = st.text(
    alphabet=st.characters(blacklist_categories=_SURROGATE_CATEGORIES),  # type: ignore[arg-type]
    min_size=0,
    max_size=10_000,
)

# Priority levels
priority_strategy: st.SearchStrategy[Priority] = st.sampled_from([1, 2, 3])


# === Property Tests ===


@given(
    total=budget_strategy,
    reserved=reserved_strategy,
    contents=st.lists(st.tuples(priority_strategy, text_strategy), min_size=0, max_size=10),
)
@settings(max_examples=200)
def test_budget_invariant(
    total: int,
    reserved: int,
    contents: list[tuple[Priority, str]],
) -> None:
    """Total allocated tokens never exceed total_budget - reserved_budget."""
    assume(reserved <= total)  # Skip invalid configs

    manager = TokenBudgetManager(
        total_budget=total,
        reserved_budget=reserved,
        model_context_limit=200_000,
    )

    for priority, content in contents:
        manager.allocate(priority, content)

    # Core invariant: used tokens <= available budget
    assert manager._used_tokens <= total - reserved


@given(
    content=st.text(
        alphabet=st.characters(blacklist_categories=_SURROGATE_CATEGORIES),  # type: ignore[arg-type]
        min_size=0,
        max_size=5000,
    ),
)
@settings(max_examples=200)
def test_unicode_safety(content: str) -> None:
    """Manager handles arbitrary unicode without crashing."""
    manager = TokenBudgetManager(total_budget=100_000, reserved_budget=0)

    result = manager.allocate(1, content)

    # Should return string (possibly truncated or empty)
    assert isinstance(result, str)
    # Remaining should be non-negative
    assert manager.remaining() >= 0


def test_empty_string_returns_empty() -> None:
    """Empty input returns empty output without consuming budget."""
    manager = TokenBudgetManager(total_budget=1000, reserved_budget=0)
    initial_remaining = manager.remaining()

    result = manager.allocate(1, "")

    assert result == ""
    assert manager.remaining() == initial_remaining


@given(content=text_strategy)
@settings(max_examples=50)
def test_zero_budget_returns_empty(content: str) -> None:
    """Zero available budget returns empty string."""
    manager = TokenBudgetManager(total_budget=100, reserved_budget=100)

    result = manager.allocate(1, content)

    assert result == ""
    assert manager._used_tokens == 0


@given(
    total=st.integers(min_value=100, max_value=10_000),
    p1_content=st.text(min_size=10, max_size=100),
    p3_content=st.text(min_size=10, max_size=100),
)
@settings(max_examples=100)
def test_allocation_tracking_by_priority(
    total: int,
    p1_content: str,
    p3_content: str,
) -> None:
    """Allocations are correctly tracked per priority level."""
    manager = TokenBudgetManager(total_budget=total, reserved_budget=0)

    manager.allocate(3, p3_content)
    p3_allocated = manager._allocations[3]

    manager.allocate(1, p1_content)
    p1_allocated = manager._allocations[1]

    # Allocations should be tracked separately
    assert manager._allocations[1] == p1_allocated
    assert manager._allocations[3] == p3_allocated
    # Total used should equal sum of priorities
    assert manager._used_tokens == sum(manager._allocations.values())


@given(
    total=st.integers(min_value=1, max_value=100_000),
    reserved=st.integers(min_value=0, max_value=100_000),
    content_size=st.integers(min_value=0, max_value=5000),
)
@settings(max_examples=100, deadline=None)  # Disable deadline for token counting
def test_remaining_never_negative(total: int, reserved: int, content_size: int) -> None:
    """remaining() is always >= 0 regardless of input."""
    assume(reserved <= total)

    manager = TokenBudgetManager(total_budget=total, reserved_budget=reserved)

    # Before any allocation
    assert manager.remaining() >= 0

    # After allocating content (possibly more than budget)
    manager.allocate(1, "x" * content_size)
    assert manager.remaining() >= 0


@given(
    total=budget_strategy,
    reserved=reserved_strategy,
)
@settings(max_examples=50)
def test_initial_remaining_equals_available(total: int, reserved: int) -> None:
    """Initial remaining() equals total_budget - reserved_budget."""
    assume(reserved <= total)

    manager = TokenBudgetManager(total_budget=total, reserved_budget=reserved)

    assert manager.remaining() == total - reserved


@given(content=text_strategy)
@settings(max_examples=100)
def test_allocate_returns_substring_or_empty(content: str) -> None:
    """Allocated content is either empty, the original, or a truncated version."""
    manager = TokenBudgetManager(total_budget=1000, reserved_budget=0)

    result = manager.allocate(1, content)

    # Result should be empty, identical, or a prefix (possibly with truncation marker)
    if result and not result.endswith("[...truncated]"):
        assert result == content or content.startswith(result.rstrip())


@given(
    total=st.integers(min_value=0, max_value=1000),
    contents=st.lists(text_strategy, min_size=1, max_size=5),
)
@settings(max_examples=100)
def test_summary_matches_allocations(
    total: int,
    contents: list[str],
) -> None:
    """Summary dict accurately reflects internal allocation state."""
    manager = TokenBudgetManager(total_budget=total, reserved_budget=0)

    priorities: tuple[Priority, Priority, Priority] = (1, 2, 3)
    for i, content in enumerate(contents):
        priority = priorities[i % 3]
        manager.allocate(priority, content)

    summary = manager.summary()

    assert summary["priority_1"] == manager._allocations[1]
    assert summary["priority_2"] == manager._allocations[2]
    assert summary["priority_3"] == manager._allocations[3]
