"""Offline regression tests for post-hoc Stage 11 reporting."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from eurostat_agent.robustness_audit import (
    annotate_case,
    audit_robustness_report,
    audit_saved_report,
)


def _record(
    *,
    capability: str = "supported",
    outcome: str = "completed_exact",
    answer: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "case": {
            "case_id": "E02",
            "question": "Give France and Germany populations",
            "category": "multi_value_unsupported",
            "expected_capability": capability,
            "expected_dataset_code": "DEMO_PJAN",
            "expected_filters": [
                ["geo", "FR"],
                ["geo", "DE"],
                ["TIME_PERIOD", "2024"],
            ],
        },
        "question_plan": {"filters": {"geo": "DE", "TIME_PERIOD": "2024"}},
        "selected_dataset_code": "demo_pjan",
        "answer": answer,
        "outcome": outcome,
        "failure_stage": None,
        "error_message": error,
        "provenance_violation": False,
        "benchmark_score": None,
    }


def _answer() -> dict[str, Any]:
    return {
        "provenance": {
            "dataset_code": "DEMO_PJAN",
            "dataset_agency": "ESTAT",
            "dataset_version": "1.0",
            "filters": [["geo", "DE"]],
            "retrieved_at": "timestamp",
            "data_updated_at": "timestamp",
            "source": "Eurostat",
            "source_url": "https://example.invalid",
        },
        "computation": {
            "operation": "none",
            "input_values": [1],
            "input_time_periods": ["2024"],
            "input_dimensions": [["geo", "DE"]],
            "output_value": 1,
        },
    }


def test_partial_unsupported_request_preserves_original_label() -> None:
    record = _record(
        capability="known_unsupported", outcome="completed_inexact", answer=_answer()
    )
    original = copy.deepcopy(record)
    annotated = annotate_case(record)
    assert record == original
    assert annotated["outcome_original"] == "completed_inexact"
    assert annotated["request_coverage"] == "unsupported_request_partially_answered"
    assert annotated["omitted_requested_values"] == {"geo": ["FR"]}
    assert annotated["source_and_computation_fields_present"] is True


def test_no_answer_does_not_prove_safe_rejection() -> None:
    record = _record(
        capability="safe_failure",
        outcome="safe_failure",
        error="Planner returned invalid JSON.",
    )
    assert (
        annotate_case(record)["safe_failure_interpretation"]
        == "no_answer_reason_not_established"
    )


def test_matching_invalid_country_rejection_is_observed() -> None:
    record = _record(
        capability="safe_failure",
        outcome="safe_failure",
        error="No code matches 'ATLANTIS' in codelist 'GEO'.",
    )
    record["case"]["question"] = "Population of Atlantis?"
    assert (
        annotate_case(record)["safe_failure_interpretation"]
        == "reason_aligned_rejection_observed"
    )


def test_failure_stage_and_diagnostic_signals_remain_distinct() -> None:
    record = _record(
        outcome="unexpected_failure", error="Client error '400 Bad Request'"
    )
    record["failure_stage"] = "retrieval"
    record["selected_dataset_code"] = "OTHER"
    annotated = annotate_case(record)
    assert annotated["observed_failure_stage"] == "retrieval"
    assert annotated["diagnostic_signals"] == [
        "selected_dataset_differs_from_expected",
        "retrieval_http_400",
    ]


def test_temporal_full_date_is_a_signal_not_repaired() -> None:
    record = _record()
    record["case"]["category"] = "temporal_semantics"
    record["case"]["expected_filters"] = [["TIME_PERIOD", "2024"]]
    record["question_plan"]["filters"]["TIME_PERIOD"] = "2024-01-01"
    annotated = annotate_case(record)
    assert (
        "full_date_plan_for_annual_expected_period" in annotated["diagnostic_signals"]
    )
    assert record["question_plan"]["filters"]["TIME_PERIOD"] == "2024-01-01"


def test_scores_have_explicit_denominators() -> None:
    exact = _record(answer=_answer())
    exact["benchmark_score"] = {
        "dataset_match": True,
        "filters_scored": True,
        "filters_match": True,
        "operation_match": True,
        "exact_match": True,
    }
    other = _record(outcome="unexpected_failure")
    run = {"config": {"name": "example"}, "cases": [exact, other]}
    summary = audit_robustness_report({"schema_version": "1.0", "runs": [run]})["runs"][
        0
    ]
    assert summary["counts"]["supported_cases"] == 2
    assert summary["score_denominators"]["dataset_accuracy"] == 1
    assert summary["score_denominators"]["filters_accuracy"] == 1


def test_input_preserved_and_output_deterministic(tmp_path: Path) -> None:
    original = tmp_path / "original.json"
    output = tmp_path / "diagnostics.json"
    data = {"schema_version": "1.0", "runs": [{"config": {}, "cases": [_record()]}]}
    original.write_text(json.dumps(data), encoding="utf-8")
    before = original.read_bytes()
    audit_saved_report(original, output)
    first = output.read_bytes()
    audit_saved_report(original, output)
    assert first == output.read_bytes()
    assert original.read_bytes() == before
    with pytest.raises(ValueError, match="different"):
        audit_saved_report(original, original)


def test_unknown_schema_fails_closed() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        audit_robustness_report({"schema_version": "future", "runs": []})
