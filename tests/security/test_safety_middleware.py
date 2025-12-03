"""Tests for the safety middleware in core/safety.py."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from jpscripts.core.cost_tracker import TokenUsage
from jpscripts.core.errors import SecurityError
from jpscripts.core.runtime import CircuitBreaker, runtime_context
from jpscripts.core.safety import (
    check_circuit_breaker,
    estimate_tokens_from_args,
    guarded_execution,
    wrap_with_breaker,
)


class TestEstimateTokensFromArgs:
    """Tests for estimate_tokens_from_args function."""

    def test_empty_args_returns_minimum(self) -> None:
        """Empty arguments should return minimum of 1 token."""
        assert estimate_tokens_from_args() == 1

    def test_string_estimation(self) -> None:
        """Strings should be estimated at ~4 chars per token."""
        # 40 chars -> ~10 tokens
        result = estimate_tokens_from_args("a" * 40)
        assert result == 10

    def test_short_string_minimum(self) -> None:
        """Short strings should return at least 1 token."""
        result = estimate_tokens_from_args("hi")
        assert result >= 1

    def test_dict_estimation(self) -> None:
        """Dicts should sum key + value tokens."""
        result = estimate_tokens_from_args(data={"key": "value"})
        # "key" -> 1, "value" -> 1+, should be > 1
        assert result >= 2

    def test_nested_structures(self) -> None:
        """Nested structures should recurse correctly."""
        nested = {"items": ["a" * 20, "b" * 20], "count": 2}
        result = estimate_tokens_from_args(nested)
        # Should be reasonable estimate
        assert result >= 10

    def test_path_estimation(self) -> None:
        """Path objects should estimate from string representation."""
        result = estimate_tokens_from_args(path=Path("/usr/local/bin/python"))
        assert result >= 5

    def test_primitives_return_one(self) -> None:
        """Primitives (int, float, bool, None) should return 1 token each."""
        result = estimate_tokens_from_args(42, 3.14, True, None)
        assert result == 4


class TestCheckCircuitBreaker:
    """Tests for check_circuit_breaker function."""

    def test_healthy_breaker_passes(self) -> None:
        """A healthy circuit breaker should not raise."""
        breaker = CircuitBreaker(
            max_cost_velocity=Decimal("10.0"),
            max_file_churn=20,
            model_id="test",
        )
        usage = TokenUsage(prompt_tokens=100, completion_tokens=0)

        # Should not raise
        check_circuit_breaker(
            breaker,
            usage,
            files_touched=[],
            persona="test",
            context="test_op",
        )

    def test_tripped_breaker_raises_security_error(self) -> None:
        """A tripped circuit breaker should raise SecurityError."""
        # Create a breaker with very low thresholds
        breaker = CircuitBreaker(
            max_cost_velocity=Decimal("0.00001"),  # Very low
            max_file_churn=0,
            model_id="test",
        )
        # First call sets the timestamp
        breaker.check_health(TokenUsage(prompt_tokens=0, completion_tokens=0), [])

        usage = TokenUsage(prompt_tokens=100000, completion_tokens=100000)

        with pytest.raises(SecurityError) as exc_info:
            check_circuit_breaker(
                breaker,
                usage,
                files_touched=[],
                persona="test",
                context="test_op",
            )

        assert "Circuit breaker tripped" in str(exc_info.value)
        assert "persona=test" in str(exc_info.value)

    def test_file_churn_trips_breaker(self) -> None:
        """Exceeding file churn threshold should trip the breaker."""
        breaker = CircuitBreaker(
            max_cost_velocity=Decimal("100.0"),
            max_file_churn=2,  # Low threshold
            model_id="test",
        )
        usage = TokenUsage(prompt_tokens=1, completion_tokens=0)

        with pytest.raises(SecurityError) as exc_info:
            check_circuit_breaker(
                breaker,
                usage,
                files_touched=[Path("a.py"), Path("b.py"), Path("c.py")],
                persona="test",
                context="too_many_files",
            )

        assert "File churn" in str(exc_info.value) or "file_churn" in str(exc_info.value)


class TestGuardedExecution:
    """Tests for guarded_execution decorator."""

    @pytest.fixture
    def test_config(self, tmp_path: Path):
        """Create a test config for runtime context."""
        from jpscripts.core.config import AppConfig, UserConfig

        return AppConfig(
            user=UserConfig(
                workspace_root=tmp_path,
                notes_dir=tmp_path / "notes",
            )
        )

    def test_decorated_function_executes_normally(self, test_config) -> None:
        """A guarded function should execute when breaker is healthy."""

        @guarded_execution("test", "test_tool")
        async def my_tool(message: str) -> str:
            return f"Hello, {message}!"

        import asyncio

        with runtime_context(test_config, workspace=test_config.user.workspace_root):
            result = asyncio.run(my_tool("world"))

        assert result == "Hello, world!"

    def test_decorated_function_preserves_metadata(self) -> None:
        """Decorator should preserve function name and docstring."""

        @guarded_execution("test", "test_tool")
        async def my_documented_tool(x: int) -> int:
            """This is the docstring."""
            return x * 2

        assert my_documented_tool.__name__ == "my_documented_tool"
        assert "docstring" in (my_documented_tool.__doc__ or "")

    def test_explicit_breaker_is_used(self) -> None:
        """When a breaker is provided, it should be used instead of runtime's."""
        breaker = CircuitBreaker(
            max_cost_velocity=Decimal("0.000001"),  # Will trip
            max_file_churn=0,
            model_id="test",
        )
        # Prime the breaker
        breaker.check_health(TokenUsage(prompt_tokens=0, completion_tokens=0), [])

        @guarded_execution("test", "test_tool", breaker=breaker)
        async def my_tool(data: str) -> str:
            return data

        import asyncio

        with pytest.raises(SecurityError):
            asyncio.run(my_tool("x" * 10000))


class TestWrapWithBreaker:
    """Tests for wrap_with_breaker function."""

    def test_wraps_function_with_enforcement(self) -> None:
        """wrap_with_breaker should add circuit breaker checks."""
        call_count = 0

        async def original_func(message: str) -> str:
            nonlocal call_count
            call_count += 1
            return message

        breaker = CircuitBreaker(
            max_cost_velocity=Decimal("100.0"),
            max_file_churn=20,
            model_id="test",
        )

        wrapped = wrap_with_breaker(original_func, "mcp", "tool:test", breaker)

        import asyncio

        result = asyncio.run(wrapped("hello"))

        assert result == "hello"
        assert call_count == 1

    def test_wrapped_function_preserves_name(self) -> None:
        """Wrapped function should keep original name."""

        async def my_named_function(x: str) -> str:
            return x

        breaker = CircuitBreaker(
            max_cost_velocity=Decimal("100.0"),
            max_file_churn=20,
            model_id="test",
        )

        wrapped = wrap_with_breaker(my_named_function, "mcp", "tool:test", breaker)

        assert wrapped.__name__ == "my_named_function"

    def test_wrapped_function_trips_on_high_cost(self) -> None:
        """Wrapped function should raise SecurityError when breaker trips."""

        async def expensive_func(data: str) -> str:
            return data

        breaker = CircuitBreaker(
            max_cost_velocity=Decimal("0.000001"),
            max_file_churn=0,
            model_id="test",
        )
        # Prime the timestamp
        breaker.check_health(TokenUsage(prompt_tokens=0, completion_tokens=0), [])

        wrapped = wrap_with_breaker(expensive_func, "mcp", "tool:expensive", breaker)

        import asyncio

        with pytest.raises(SecurityError):
            asyncio.run(wrapped("x" * 100000))
