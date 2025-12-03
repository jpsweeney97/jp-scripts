"""Tests for CircuitBreaker thread safety and SharedRateLimiter."""

from __future__ import annotations

import asyncio
import threading
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from jpscripts.core.cost_tracker import TokenUsage
from jpscripts.core.runtime import CircuitBreaker
from jpscripts.core.security.rate_limit import (
    SharedRateLimiter,
    get_shared_rate_limiter,
    reset_shared_rate_limiter,
)

if TYPE_CHECKING:
    pass


class TestCircuitBreakerThreadSafety:
    """Test thread safety of CircuitBreaker."""

    def test_concurrent_check_health_no_race(self) -> None:
        """Multiple threads calling check_health don't corrupt state."""
        breaker = CircuitBreaker(
            max_cost_velocity=Decimal("100.0"),
            max_file_churn=100,
        )

        results: list[bool] = []
        errors: list[Exception] = []

        def worker() -> None:
            try:
                usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
                files = [Path(f"/tmp/file_{i}.py") for i in range(3)]
                result = breaker.check_health(usage, files)
                results.append(result)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent access: {errors}"
        assert len(results) == 20
        # All should pass since we're under limits
        assert all(results)

    def test_lock_prevents_interleaved_state(self) -> None:
        """Verify that state updates are atomic within check_health."""
        breaker = CircuitBreaker(
            max_cost_velocity=Decimal("1.0"),
            max_file_churn=5,
        )

        # First call establishes baseline
        usage1 = TokenUsage(prompt_tokens=1000, completion_tokens=500)
        breaker.check_health(usage1, [Path("/tmp/a.py")])

        # Concurrent calls should each see consistent state
        observed_velocities: list[Decimal] = []

        def rapid_check() -> None:
            for _ in range(10):
                usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
                breaker.check_health(usage, [Path("/tmp/b.py")])
                observed_velocities.append(breaker.last_cost_velocity)

        threads = [threading.Thread(target=rapid_check) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All observed velocities should be valid Decimals (not corrupted)
        assert len(observed_velocities) == 50
        for v in observed_velocities:
            assert isinstance(v, Decimal)
            assert v >= Decimal("0")


class TestSharedRateLimiter:
    """Test SharedRateLimiter for swarm mode."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self) -> None:
        """Reset singleton before each test."""
        reset_shared_rate_limiter()
        yield
        reset_shared_rate_limiter()

    def test_singleton_creation(self) -> None:
        """get_shared_rate_limiter returns same instance."""
        limiter1 = get_shared_rate_limiter()
        limiter2 = get_shared_rate_limiter()
        assert limiter1 is limiter2

    def test_singleton_ignores_later_config(self) -> None:
        """Config parameters only apply on first creation."""
        limiter1 = get_shared_rate_limiter(max_calls=100, window_seconds=30.0)
        limiter2 = get_shared_rate_limiter(max_calls=999, window_seconds=999.0)

        assert limiter1 is limiter2
        assert limiter1.max_calls == 100
        assert limiter1.window_seconds == 30.0

    @pytest.mark.asyncio
    async def test_acquire_basic(self) -> None:
        """Basic acquire works."""
        limiter = get_shared_rate_limiter(max_calls=10, window_seconds=60.0)
        result = await limiter.acquire()
        assert result is True

    @pytest.mark.asyncio
    async def test_rate_limiting_works(self) -> None:
        """Limiter rejects when limit exceeded."""
        limiter = get_shared_rate_limiter(max_calls=3, window_seconds=60.0)

        # Exhaust the limit
        assert await limiter.acquire()
        assert await limiter.acquire()
        assert await limiter.acquire()

        # Fourth should be rejected
        assert not await limiter.acquire()
        assert limiter.is_rate_limited()

    @pytest.mark.asyncio
    async def test_reset_clears_state(self) -> None:
        """reset() clears the limiter state."""
        limiter = get_shared_rate_limiter(max_calls=2, window_seconds=60.0)

        assert await limiter.acquire()
        assert await limiter.acquire()
        assert not await limiter.acquire()

        limiter.reset()

        # Should be able to acquire again
        assert await limiter.acquire()

    def test_thread_safe_singleton_creation(self) -> None:
        """Multiple threads creating singleton get same instance."""
        results: list[SharedRateLimiter] = []

        def get_limiter() -> None:
            results.append(get_shared_rate_limiter())

        threads = [threading.Thread(target=get_limiter) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should be the same instance
        assert len(results) == 20
        first = results[0]
        assert all(r is first for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_acquires(self) -> None:
        """Concurrent acquire calls are properly synchronized."""
        limiter = get_shared_rate_limiter(max_calls=50, window_seconds=60.0)

        async def acquire_many(count: int) -> int:
            successes = 0
            for _ in range(count):
                if await limiter.acquire():
                    successes += 1
            return successes

        # Run 10 concurrent tasks each trying to acquire 10 times
        tasks = [acquire_many(10) for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # Total successes should be exactly 50 (the limit)
        assert sum(results) == 50

    def test_current_usage(self) -> None:
        """current_usage returns correct values."""
        limiter = SharedRateLimiter(max_calls=10, window_seconds=60.0)
        used, max_calls = limiter.current_usage()
        assert used == 0
        assert max_calls == 10

    def test_time_until_available(self) -> None:
        """time_until_available returns 0 when not limited."""
        limiter = SharedRateLimiter(max_calls=10, window_seconds=60.0)
        assert limiter.time_until_available() == 0.0
