"""
Rate limiting for shell command execution.

This module provides a token bucket rate limiter to prevent runaway
agent loops from executing unlimited shell commands.

Usage:
    limiter = RateLimiter(max_calls=100, window_seconds=60.0)

    if await limiter.acquire():
        # Execute command
        pass
    else:
        # Rate limited - wait or fail

    # Or block until available:
    await limiter.wait_and_acquire()
    # Execute command
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from time import monotonic
from typing import Deque


@dataclass
class RateLimiter:
    """Token bucket rate limiter for shell commands.

    Uses a sliding window approach: tracks timestamps of recent calls
    and rejects new calls if too many occurred within the window.

    Attributes:
        max_calls: Maximum number of calls allowed per window
        window_seconds: Duration of the sliding window in seconds
    """

    max_calls: int = 100
    window_seconds: float = 60.0
    _timestamps: Deque[float] = field(default_factory=deque)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        if self.max_calls <= 0:
            raise ValueError("max_calls must be positive")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")

    def _prune_old_timestamps(self, now: float) -> None:
        """Remove timestamps that have fallen outside the window."""
        cutoff = now - self.window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    async def acquire(self) -> bool:
        """Attempt to acquire permission to make a call.

        This is a non-blocking operation that immediately returns
        whether the call is allowed.

        Returns:
            True if the call is allowed, False if rate limited
        """
        async with self._lock:
            now = monotonic()
            self._prune_old_timestamps(now)

            if len(self._timestamps) >= self.max_calls:
                return False

            self._timestamps.append(now)
            return True

    async def wait_and_acquire(self, timeout: float | None = None) -> bool:
        """Wait until a call can be made, then acquire.

        Blocks until rate limiting allows the call, or until timeout.

        Args:
            timeout: Maximum time to wait in seconds (None = wait forever)

        Returns:
            True if acquired, False if timeout expired
        """
        start = monotonic()
        while True:
            if await self.acquire():
                return True

            # Check timeout
            if timeout is not None:
                elapsed = monotonic() - start
                if elapsed >= timeout:
                    return False

            # Calculate time until oldest entry expires
            async with self._lock:
                if self._timestamps:
                    now = monotonic()
                    oldest = self._timestamps[0]
                    wait_time = max(0.01, (oldest + self.window_seconds) - now)
                else:
                    wait_time = 0.01

            # Don't wait longer than remaining timeout
            if timeout is not None:
                elapsed = monotonic() - start
                remaining = timeout - elapsed
                wait_time = min(wait_time, remaining)
                if wait_time <= 0:
                    return False

            await asyncio.sleep(wait_time)

    def current_usage(self) -> tuple[int, int]:
        """Return current usage as (calls_made, max_calls).

        Note: This is a point-in-time snapshot and may be stale immediately.
        """
        now = monotonic()
        self._prune_old_timestamps(now)
        return len(self._timestamps), self.max_calls

    def is_rate_limited(self) -> bool:
        """Check if currently rate limited (non-async convenience method).

        Note: This is a point-in-time check and the state may change immediately.
        """
        now = monotonic()
        self._prune_old_timestamps(now)
        return len(self._timestamps) >= self.max_calls

    def reset(self) -> None:
        """Clear all recorded timestamps, resetting the limiter."""
        self._timestamps.clear()

    def time_until_available(self) -> float:
        """Return seconds until a slot becomes available.

        Returns:
            0.0 if immediately available, otherwise seconds to wait
        """
        now = monotonic()
        self._prune_old_timestamps(now)

        if len(self._timestamps) < self.max_calls:
            return 0.0

        if not self._timestamps:
            return 0.0

        oldest = self._timestamps[0]
        wait_time = (oldest + self.window_seconds) - now
        return max(0.0, wait_time)


class RateLimitExceeded(Exception):
    """Raised when an operation is rejected due to rate limiting."""

    def __init__(self, wait_seconds: float) -> None:
        self.wait_seconds = wait_seconds
        super().__init__(
            f"Rate limit exceeded. Try again in {wait_seconds:.1f} seconds."
        )


async def rate_limited_call(
    limiter: RateLimiter,
    coro,
    *,
    timeout: float | None = None,
    block: bool = True,
):
    """Execute a coroutine with rate limiting.

    Args:
        limiter: The RateLimiter to use
        coro: The coroutine to execute
        timeout: Maximum time to wait for rate limit (only if block=True)
        block: If True, wait for rate limit. If False, raise immediately.

    Returns:
        The result of the coroutine

    Raises:
        RateLimitExceeded: If rate limited and block=False or timeout expired
    """
    if block:
        acquired = await limiter.wait_and_acquire(timeout=timeout)
        if not acquired:
            wait_time = limiter.time_until_available()
            raise RateLimitExceeded(wait_time)
    else:
        acquired = await limiter.acquire()
        if not acquired:
            wait_time = limiter.time_until_available()
            raise RateLimitExceeded(wait_time)

    return await coro


__all__ = [
    "RateLimiter",
    "RateLimitExceeded",
    "rate_limited_call",
]
