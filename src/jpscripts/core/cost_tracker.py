"""
Token usage and cost tracking for LLM operations.

This module tracks token consumption and provides estimated costs
based on current model pricing.

Usage:
    tracker = CostTracker(model_id="claude-opus-4-5")

    # After each LLM call
    tracker.record_usage(TokenUsage(prompt_tokens=1000, completion_tokens=500))

    # Check current costs
    print(f"Estimated cost: ${tracker.estimated_cost}")

    # Enforce budget limits
    if not tracker.check_budget():
        raise BudgetExceeded(tracker.estimated_cost, tracker.budget_limit)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token usage for a single LLM request."""

    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# Pricing per 1M tokens (as of 2024-2025)
# Updated periodically - these are estimates
MODEL_PRICING: dict[str, dict[str, Decimal]] = {
    # Anthropic Claude models
    "claude-opus-4-5": {"input": Decimal("15.00"), "output": Decimal("75.00")},
    "claude-sonnet-4-5": {"input": Decimal("3.00"), "output": Decimal("15.00")},
    "claude-sonnet-4": {"input": Decimal("3.00"), "output": Decimal("15.00")},
    "claude-haiku-3-5": {"input": Decimal("0.80"), "output": Decimal("4.00")},
    "claude-3-opus": {"input": Decimal("15.00"), "output": Decimal("75.00")},
    "claude-3-sonnet": {"input": Decimal("3.00"), "output": Decimal("15.00")},
    "claude-3-haiku": {"input": Decimal("0.25"), "output": Decimal("1.25")},
    # OpenAI models
    "gpt-4-turbo": {"input": Decimal("10.00"), "output": Decimal("30.00")},
    "gpt-4o": {"input": Decimal("5.00"), "output": Decimal("15.00")},
    "gpt-4o-mini": {"input": Decimal("0.15"), "output": Decimal("0.60")},
    "o1": {"input": Decimal("15.00"), "output": Decimal("60.00")},
    "o1-mini": {"input": Decimal("3.00"), "output": Decimal("12.00")},
    # Default fallback
    "default": {"input": Decimal("10.00"), "output": Decimal("30.00")},
}


def _normalize_model_id(model_id: str) -> str:
    """Normalize model ID for pricing lookup."""
    normalized = model_id.lower().strip()
    # Handle common variations
    if normalized.startswith("claude-opus-4"):
        return "claude-opus-4-5"
    if normalized.startswith("claude-sonnet-4"):
        return "claude-sonnet-4"
    if normalized.startswith("gpt-4-turbo"):
        return "gpt-4-turbo"
    return normalized


def get_pricing(model_id: str) -> dict[str, Decimal]:
    """Get pricing for a model, falling back to default if unknown."""
    normalized = _normalize_model_id(model_id)
    return MODEL_PRICING.get(normalized, MODEL_PRICING["default"])


@dataclass
class CostTracker:
    """Track token usage and estimated costs across requests.

    Attributes:
        model_id: The model being used (for pricing lookup)
        budget_limit: Optional maximum cost in USD (None = no limit)
    """

    model_id: str = "claude-opus-4-5"
    budget_limit: Decimal | None = None

    total_input_tokens: int = field(default=0, init=False)
    total_output_tokens: int = field(default=0, init=False)
    request_count: int = field(default=0, init=False)
    _usage_history: list[TokenUsage] = field(default_factory=lambda: [], init=False)

    def record_usage(self, usage: TokenUsage) -> None:
        """Record token usage from a request."""
        self.total_input_tokens += usage.prompt_tokens
        self.total_output_tokens += usage.completion_tokens
        self.request_count += 1
        self._usage_history.append(usage)

    def record_from_dict(self, data: dict[str, Any]) -> None:
        """Record usage from a dict (e.g., from API response)."""
        prompt = data.get("prompt_tokens", data.get("input_tokens", 0))
        completion = data.get("completion_tokens", data.get("output_tokens", 0))
        self.record_usage(TokenUsage(prompt_tokens=prompt, completion_tokens=completion))

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def estimated_cost(self) -> Decimal:
        """Calculate estimated cost in USD."""
        pricing = get_pricing(self.model_id)
        input_cost = (Decimal(self.total_input_tokens) / Decimal(1_000_000)) * pricing["input"]
        output_cost = (Decimal(self.total_output_tokens) / Decimal(1_000_000)) * pricing["output"]
        return input_cost + output_cost

    @property
    def estimated_cost_formatted(self) -> str:
        """Return estimated cost as formatted string."""
        cost = self.estimated_cost
        if cost < Decimal("0.01"):
            return f"${cost:.4f}"
        return f"${cost:.2f}"

    def check_budget(self) -> bool:
        """Check if current cost is within budget.

        Returns:
            True if within budget (or no budget set), False if exceeded
        """
        if self.budget_limit is None:
            return True
        return self.estimated_cost < self.budget_limit

    def remaining_budget(self) -> Decimal | None:
        """Return remaining budget, or None if no limit set."""
        if self.budget_limit is None:
            return None
        return max(Decimal(0), self.budget_limit - self.estimated_cost)

    def average_tokens_per_request(self) -> tuple[float, float]:
        """Return average (input, output) tokens per request."""
        if self.request_count == 0:
            return 0.0, 0.0
        avg_input = self.total_input_tokens / self.request_count
        avg_output = self.total_output_tokens / self.request_count
        return avg_input, avg_output

    def summary(self) -> dict[str, Any]:
        """Return a summary dict for logging/display."""
        return {
            "model": self.model_id,
            "requests": self.request_count,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": str(self.estimated_cost),
            "budget_limit_usd": str(self.budget_limit) if self.budget_limit else None,
            "within_budget": self.check_budget(),
        }

    def reset(self) -> None:
        """Reset all tracking counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.request_count = 0
        self._usage_history.clear()


class BudgetExceeded(Exception):
    """Raised when operation would exceed the budget limit."""

    def __init__(self, current_cost: Decimal, budget_limit: Decimal) -> None:
        self.current_cost = current_cost
        self.budget_limit = budget_limit
        super().__init__(f"Budget exceeded: ${current_cost:.4f} >= ${budget_limit:.4f}")


__all__ = [
    "MODEL_PRICING",
    "BudgetExceeded",
    "CostTracker",
    "TokenUsage",
    "get_pricing",
]
