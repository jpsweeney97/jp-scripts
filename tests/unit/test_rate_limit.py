"""Tests for core/rate_limit.py - token bucket rate limiter."""

from __future__ import annotations

import asyncio
from time import monotonic

import pytest

from jpscripts.core.rate_limit import (
    RateLimiter,
    RateLimitExceeded,
    rate_limited_call,
)

# ---------------------------------------------------------------------------
# Test RateLimiter initialization validation
# ---------------------------------------------------------------------------


class TestRateLimiterInit:
    """Test RateLimiter initialization and validation."""

    def test_valid_init(self) -> None:
        limiter = RateLimiter(max_calls=10, window_seconds=5.0)
        assert limiter.max_calls == 10
        assert limiter.window_seconds == 5.0

    def test_max_calls_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_calls must be positive"):
            RateLimiter(max_calls=0, window_seconds=1.0)

    def test_max_calls_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="max_calls must be positive"):
            RateLimiter(max_calls=-1, window_seconds=1.0)

    def test_window_seconds_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="window_seconds must be positive"):
            RateLimiter(max_calls=10, window_seconds=0.0)

    def test_window_seconds_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="window_seconds must be positive"):
            RateLimiter(max_calls=10, window_seconds=-1.0)

    def test_default_values(self) -> None:
        limiter = RateLimiter()
        assert limiter.max_calls == 100
        assert limiter.window_seconds == 60.0


# ---------------------------------------------------------------------------
# Test acquire
# ---------------------------------------------------------------------------


class TestAcquire:
    """Test the acquire method."""

    @pytest.mark.asyncio
    async def test_acquire_allows_within_limit(self) -> None:
        limiter = RateLimiter(max_calls=5, window_seconds=60.0)
        for _ in range(5):
            result = await limiter.acquire()
            assert result is True

    @pytest.mark.asyncio
    async def test_acquire_rejects_at_limit(self) -> None:
        limiter = RateLimiter(max_calls=3, window_seconds=60.0)
        # Fill up the limit
        for _ in range(3):
            await limiter.acquire()
        # Next should be rejected
        result = await limiter.acquire()
        assert result is False

    @pytest.mark.asyncio
    async def test_acquire_records_timestamp(self) -> None:
        limiter = RateLimiter(max_calls=10, window_seconds=60.0)
        await limiter.acquire()
        assert len(limiter._timestamps) == 1


# ---------------------------------------------------------------------------
# Test timestamp expiration
# ---------------------------------------------------------------------------


class TestTimestampExpiration:
    """Test that old timestamps expire correctly."""

    @pytest.mark.asyncio
    async def test_timestamps_expire_after_window(self) -> None:
        limiter = RateLimiter(max_calls=2, window_seconds=0.1)
        # Fill the limit
        await limiter.acquire()
        await limiter.acquire()
        # Wait for window to pass
        await asyncio.sleep(0.15)
        # Should be able to acquire again
        result = await limiter.acquire()
        assert result is True

    def test_prune_old_timestamps(self) -> None:
        limiter = RateLimiter(max_calls=10, window_seconds=1.0)
        now = monotonic()
        # Add timestamps: one old, one recent
        limiter._timestamps.append(now - 2.0)  # Old
        limiter._timestamps.append(now - 0.5)  # Recent
        limiter._prune_old_timestamps(now)
        assert len(limiter._timestamps) == 1


# ---------------------------------------------------------------------------
# Test wait_and_acquire
# ---------------------------------------------------------------------------


class TestWaitAndAcquire:
    """Test the blocking wait_and_acquire method."""

    @pytest.mark.asyncio
    async def test_wait_and_acquire_immediate_success(self) -> None:
        limiter = RateLimiter(max_calls=5, window_seconds=60.0)
        result = await limiter.wait_and_acquire()
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_and_acquire_blocks_until_available(self) -> None:
        limiter = RateLimiter(max_calls=1, window_seconds=0.1)
        await limiter.acquire()  # Fill the limit

        start = monotonic()
        result = await limiter.wait_and_acquire()
        elapsed = monotonic() - start

        assert result is True
        assert elapsed >= 0.08  # Should have waited close to window time

    @pytest.mark.asyncio
    async def test_wait_and_acquire_timeout_returns_false(self) -> None:
        limiter = RateLimiter(max_calls=1, window_seconds=10.0)
        await limiter.acquire()  # Fill the limit

        result = await limiter.wait_and_acquire(timeout=0.05)
        assert result is False

    @pytest.mark.asyncio
    async def test_wait_and_acquire_timeout_respects_limit(self) -> None:
        limiter = RateLimiter(max_calls=1, window_seconds=10.0)
        await limiter.acquire()

        start = monotonic()
        await limiter.wait_and_acquire(timeout=0.1)
        elapsed = monotonic() - start

        # Should not wait much longer than timeout
        assert elapsed < 0.3

    @pytest.mark.asyncio
    async def test_wait_and_acquire_no_timeout_waits_indefinitely(self) -> None:
        limiter = RateLimiter(max_calls=1, window_seconds=0.05)
        await limiter.acquire()

        # Should eventually succeed without timeout
        result = await limiter.wait_and_acquire(timeout=None)
        assert result is True


# ---------------------------------------------------------------------------
# Test current_usage
# ---------------------------------------------------------------------------


class TestCurrentUsage:
    """Test the current_usage method."""

    @pytest.mark.asyncio
    async def test_current_usage_returns_tuple(self) -> None:
        limiter = RateLimiter(max_calls=10, window_seconds=60.0)
        usage = limiter.current_usage()
        assert isinstance(usage, tuple)
        assert len(usage) == 2

    @pytest.mark.asyncio
    async def test_current_usage_reflects_calls(self) -> None:
        limiter = RateLimiter(max_calls=10, window_seconds=60.0)
        assert limiter.current_usage() == (0, 10)
        await limiter.acquire()
        assert limiter.current_usage() == (1, 10)
        await limiter.acquire()
        assert limiter.current_usage() == (2, 10)

    @pytest.mark.asyncio
    async def test_current_usage_prunes_old(self) -> None:
        limiter = RateLimiter(max_calls=10, window_seconds=0.05)
        await limiter.acquire()
        assert limiter.current_usage()[0] == 1
        await asyncio.sleep(0.1)
        assert limiter.current_usage()[0] == 0


# ---------------------------------------------------------------------------
# Test is_rate_limited
# ---------------------------------------------------------------------------


class TestIsRateLimited:
    """Test the is_rate_limited method."""

    @pytest.mark.asyncio
    async def test_is_rate_limited_false_when_under_limit(self) -> None:
        limiter = RateLimiter(max_calls=5, window_seconds=60.0)
        assert limiter.is_rate_limited() is False
        await limiter.acquire()
        assert limiter.is_rate_limited() is False

    @pytest.mark.asyncio
    async def test_is_rate_limited_true_at_limit(self) -> None:
        limiter = RateLimiter(max_calls=2, window_seconds=60.0)
        await limiter.acquire()
        await limiter.acquire()
        assert limiter.is_rate_limited() is True

    @pytest.mark.asyncio
    async def test_is_rate_limited_becomes_false_after_expiry(self) -> None:
        limiter = RateLimiter(max_calls=1, window_seconds=0.05)
        await limiter.acquire()
        assert limiter.is_rate_limited() is True
        await asyncio.sleep(0.1)
        assert limiter.is_rate_limited() is False


# ---------------------------------------------------------------------------
# Test reset
# ---------------------------------------------------------------------------


class TestReset:
    """Test the reset method."""

    @pytest.mark.asyncio
    async def test_reset_clears_timestamps(self) -> None:
        limiter = RateLimiter(max_calls=3, window_seconds=60.0)
        await limiter.acquire()
        await limiter.acquire()
        await limiter.acquire()
        assert limiter.is_rate_limited() is True

        limiter.reset()
        assert limiter.is_rate_limited() is False
        assert len(limiter._timestamps) == 0

    @pytest.mark.asyncio
    async def test_reset_allows_new_acquisitions(self) -> None:
        limiter = RateLimiter(max_calls=1, window_seconds=60.0)
        await limiter.acquire()
        assert await limiter.acquire() is False

        limiter.reset()
        assert await limiter.acquire() is True


# ---------------------------------------------------------------------------
# Test time_until_available
# ---------------------------------------------------------------------------


class TestTimeUntilAvailable:
    """Test the time_until_available method."""

    @pytest.mark.asyncio
    async def test_time_until_available_zero_when_available(self) -> None:
        limiter = RateLimiter(max_calls=5, window_seconds=60.0)
        assert limiter.time_until_available() == 0.0

    @pytest.mark.asyncio
    async def test_time_until_available_positive_when_limited(self) -> None:
        limiter = RateLimiter(max_calls=1, window_seconds=1.0)
        await limiter.acquire()
        wait_time = limiter.time_until_available()
        assert 0.5 < wait_time <= 1.0  # Should be close to window time

    @pytest.mark.asyncio
    async def test_time_until_available_decreases_over_time(self) -> None:
        limiter = RateLimiter(max_calls=1, window_seconds=1.0)
        await limiter.acquire()
        time1 = limiter.time_until_available()
        await asyncio.sleep(0.2)
        time2 = limiter.time_until_available()
        assert time2 < time1

    def test_time_until_available_empty_timestamps(self) -> None:
        limiter = RateLimiter(max_calls=5, window_seconds=60.0)
        assert limiter.time_until_available() == 0.0


# ---------------------------------------------------------------------------
# Test RateLimitExceeded exception
# ---------------------------------------------------------------------------


class TestRateLimitExceeded:
    """Test the RateLimitExceeded exception."""

    def test_exception_has_wait_seconds(self) -> None:
        exc = RateLimitExceeded(5.5)
        assert exc.wait_seconds == 5.5

    def test_exception_message_includes_wait_time(self) -> None:
        exc = RateLimitExceeded(3.2)
        assert "3.2" in str(exc)
        assert "Rate limit exceeded" in str(exc)

    def test_exception_is_exception(self) -> None:
        exc = RateLimitExceeded(1.0)
        assert isinstance(exc, Exception)


# ---------------------------------------------------------------------------
# Test rate_limited_call
# ---------------------------------------------------------------------------


class TestRateLimitedCall:
    """Test the rate_limited_call helper function."""

    @pytest.mark.asyncio
    async def test_rate_limited_call_success(self) -> None:
        limiter = RateLimiter(max_calls=10, window_seconds=60.0)

        async def my_coro() -> str:
            return "result"

        result = await rate_limited_call(limiter, my_coro())
        assert result == "result"

    @pytest.mark.asyncio
    async def test_rate_limited_call_blocking_waits(self) -> None:
        limiter = RateLimiter(max_calls=1, window_seconds=0.1)
        await limiter.acquire()

        async def my_coro() -> str:
            return "waited"

        start = monotonic()
        result = await rate_limited_call(limiter, my_coro(), block=True)
        elapsed = monotonic() - start

        assert result == "waited"
        assert elapsed >= 0.05  # Had to wait

    @pytest.mark.asyncio
    async def test_rate_limited_call_non_blocking_raises(self) -> None:
        limiter = RateLimiter(max_calls=1, window_seconds=60.0)
        await limiter.acquire()

        async def my_coro() -> str:
            return "result"

        coro = my_coro()
        try:
            with pytest.raises(RateLimitExceeded):
                await rate_limited_call(limiter, coro, block=False)
        finally:
            coro.close()  # Clean up unawaited coroutine

    @pytest.mark.asyncio
    async def test_rate_limited_call_timeout_raises(self) -> None:
        limiter = RateLimiter(max_calls=1, window_seconds=60.0)
        await limiter.acquire()

        async def my_coro() -> str:
            return "result"

        coro = my_coro()
        try:
            with pytest.raises(RateLimitExceeded):
                await rate_limited_call(limiter, coro, block=True, timeout=0.05)
        finally:
            coro.close()  # Clean up unawaited coroutine

    @pytest.mark.asyncio
    async def test_rate_limited_call_executes_coroutine(self) -> None:
        limiter = RateLimiter(max_calls=10, window_seconds=60.0)
        executed = []

        async def my_coro() -> int:
            executed.append(1)
            return 42

        result = await rate_limited_call(limiter, my_coro())
        assert result == 42
        assert len(executed) == 1


# ---------------------------------------------------------------------------
# Test concurrent access
# ---------------------------------------------------------------------------


class TestConcurrentAccess:
    """Test rate limiter under concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_acquisitions_respect_limit(self) -> None:
        limiter = RateLimiter(max_calls=5, window_seconds=60.0)

        async def try_acquire() -> bool:
            return await limiter.acquire()

        results = await asyncio.gather(*[try_acquire() for _ in range(10)])

        successes = sum(1 for r in results if r)
        assert successes == 5  # Only 5 should succeed

    @pytest.mark.asyncio
    async def test_lock_prevents_race_conditions(self) -> None:
        limiter = RateLimiter(max_calls=3, window_seconds=60.0)

        async def rapid_acquire() -> bool:
            return await limiter.acquire()

        # Run many concurrent acquisitions
        results = await asyncio.gather(*[rapid_acquire() for _ in range(100)])

        successes = sum(1 for r in results if r)
        assert successes == 3  # Lock should prevent more than limit


# ---------------------------------------------------------------------------
# Test edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_single_call_limit(self) -> None:
        limiter = RateLimiter(max_calls=1, window_seconds=60.0)
        assert await limiter.acquire() is True
        assert await limiter.acquire() is False

    @pytest.mark.asyncio
    async def test_very_short_window(self) -> None:
        limiter = RateLimiter(max_calls=1, window_seconds=0.01)
        await limiter.acquire()
        await asyncio.sleep(0.02)
        assert await limiter.acquire() is True

    def test_time_until_available_never_negative(self) -> None:
        limiter = RateLimiter(max_calls=1, window_seconds=0.001)
        # Even with tiny window, should never return negative
        limiter._timestamps.append(monotonic() - 10)  # Very old timestamp
        result = limiter.time_until_available()
        assert result >= 0.0
