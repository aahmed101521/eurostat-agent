import math

import pytest

from eurostat_agent.config import CostBudget


def test_cost_budget_preserves_limits() -> None:
    budget = CostBudget(
        max_run_usd=0.25,
        max_evaluation_usd=25.0,
    )

    assert budget.max_run_usd == 0.25
    assert budget.max_evaluation_usd == 25.0


@pytest.mark.parametrize(
    "max_run_usd,max_evaluation_usd",
    [
        (0.0, 25.0),
        (-0.01, 25.0),
        (0.25, 0.0),
        (0.25, -1.0),
    ],
)
def test_cost_budget_rejects_non_positive_limits(
    max_run_usd: float,
    max_evaluation_usd: float,
) -> None:
    with pytest.raises(ValueError):
        CostBudget(
            max_run_usd=max_run_usd,
            max_evaluation_usd=max_evaluation_usd,
        )


@pytest.mark.parametrize(
    "value",
    [
        math.inf,
        -math.inf,
        math.nan,
    ],
)
def test_cost_budget_rejects_non_finite_limits(value: float) -> None:
    with pytest.raises(ValueError):
        CostBudget(
            max_run_usd=value,
            max_evaluation_usd=25.0,
        )

    with pytest.raises(ValueError):
        CostBudget(
            max_run_usd=0.25,
            max_evaluation_usd=value,
        )
