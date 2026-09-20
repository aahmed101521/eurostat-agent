from collections import Counter

import pytest

from eurostat_agent.benchmark_cases import (
    ROBUSTNESS_BENCHMARK_CASES,
    select_robustness_cases,
)


def test_robustness_suite_is_stratified_to_36_cases() -> None:
    assert len(ROBUSTNESS_BENCHMARK_CASES) == 36
    assert len({case.case_id for case in ROBUSTNESS_BENCHMARK_CASES}) == 36

    category_counts = Counter(case.category for case in ROBUSTNESS_BENCHMARK_CASES)
    assert category_counts == {
        "direct_supported": 6,
        "linguistic_variation": 6,
        "deterministic_computation": 6,
        "temporal_semantics": 6,
        "multi_value_unsupported": 6,
        "safe_failure": 6,
    }


def test_robustness_suite_separates_capability_status() -> None:
    capability_counts = Counter(
        case.expected_capability for case in ROBUSTNESS_BENCHMARK_CASES
    )
    assert capability_counts == {
        "supported": 20,
        "known_unsupported": 10,
        "safe_failure": 6,
    }


def test_computation_cases_cover_current_operations_without_redesign() -> None:
    computation_cases = tuple(
        case
        for case in ROBUSTNESS_BENCHMARK_CASES
        if case.category == "deterministic_computation"
    )
    assert {case.expected_operation for case in computation_cases} == {
        "sum",
        "difference",
        "ratio",
        "percentage_change",
    }

    supported = tuple(
        case for case in computation_cases if case.expected_capability == "supported"
    )
    known_unsupported = tuple(
        case
        for case in computation_cases
        if case.expected_capability == "known_unsupported"
    )
    assert {case.expected_operation for case in supported} == {"sum"}
    assert {case.expected_operation for case in known_unsupported} == {
        "difference",
        "ratio",
        "percentage_change",
    }


def test_known_multi_value_cases_preserve_duplicate_dimension_expectations() -> None:
    cases = tuple(
        case
        for case in ROBUSTNESS_BENCHMARK_CASES
        if case.category == "multi_value_unsupported"
    )

    for case in cases:
        dimensions = tuple(dimension for dimension, _ in case.expected_filters)
        assert len(set(dimensions)) < len(dimensions)


def test_case_selection_preserves_requested_order() -> None:
    selected = select_robustness_cases(("F06", "A01", "D03"))
    assert tuple(case.case_id for case in selected) == ("F06", "A01", "D03")


def test_case_selection_rejects_unknown_ids() -> None:
    with pytest.raises(ValueError, match="Unknown robustness case id"):
        select_robustness_cases(("A01", "Z99"))
