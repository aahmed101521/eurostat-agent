"""Runtime configuration for eurostat-agent.

The project treats LLM cost as a first-class result rather than incidental
plumbing. Budget limits are defined here before any model integration exists,
so later agent and evaluation code cannot introduce unbounded spending by
default.

Example:
    >>> from eurostat_agent.config import CostBudget
    >>> budget = CostBudget(max_run_usd=0.25, max_evaluation_usd=25.0)
    >>> budget.max_run_usd
    0.25
"""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class CostBudget:
    """Cost limits for individual runs and complete evaluations."""

    max_run_usd: float
    max_evaluation_usd: float

    def __post_init__(self) -> None:
        """Validate that all configured limits are positive and finite."""
        for name, value in (
            ("max_run_usd", self.max_run_usd),
            ("max_evaluation_usd", self.max_evaluation_usd),
        ):
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number.")
