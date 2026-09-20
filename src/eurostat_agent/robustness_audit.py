"""Deterministic post-hoc annotations of saved Stage 11 robustness evidence.

The input run is never edited; the output is a companion report. Diagnostic
signals are observations, not independently established root causes.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_FULL_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_REQUIRED_SOURCE_FIELDS = (
    "dataset_code",
    "dataset_agency",
    "dataset_version",
    "filters",
    "retrieved_at",
    "data_updated_at",
    "source",
    "source_url",
)
_REQUIRED_COMPUTATION_FIELDS = (
    "operation",
    "input_values",
    "input_time_periods",
    "input_dimensions",
    "output_value",
)


def _is_structurally_present(answer: dict[str, Any] | None) -> bool:
    """Check field presence, not correctness of the source or numerical values."""
    if not isinstance(answer, dict):
        return False
    source = answer.get("provenance")
    computation = answer.get("computation")
    if not isinstance(source, dict) or not isinstance(computation, dict):
        return False
    return all(
        source.get(name) is not None for name in _REQUIRED_SOURCE_FIELDS
    ) and all(
        computation.get(name) is not None for name in _REQUIRED_COMPUTATION_FIELDS
    )


def _requested_multivalues(case: dict[str, Any]) -> dict[str, list[str]]:
    values: dict[str, list[str]] = defaultdict(list)
    for pair in case.get("expected_filters", []):
        if isinstance(pair, list) and len(pair) == 2:
            dimension, value = pair
            if value not in values[dimension]:
                values[dimension].append(value)
    return {key: items for key, items in sorted(values.items()) if len(items) > 1}


def annotate_case(record: dict[str, Any]) -> dict[str, Any]:
    """Return conservative, case-level annotations without modifying input."""
    case = record["case"]
    plan = record.get("question_plan")
    answer = record.get("answer")
    selected = record.get("selected_dataset_code")
    expected = case.get("expected_dataset_code")
    status = case["expected_capability"]
    outcome = record["outcome"]
    error = record.get("error_message") or ""
    signals: list[str] = []

    if record.get("failure_stage") in {"planner_model_call", "selector_model_call"}:
        signals.append("model_output_or_model_call_failure")
    if plan is not None:
        planned_period = plan.get("filters", {}).get("TIME_PERIOD")
        expected_periods = [
            pair[1]
            for pair in case.get("expected_filters", [])
            if pair[0] == "TIME_PERIOD"
        ]
        if (
            case.get("category") == "temporal_semantics"
            and isinstance(planned_period, str)
            and _FULL_DATE.fullmatch(planned_period)
            and expected_periods
            and planned_period != expected_periods[0]
        ):
            signals.append("full_date_plan_for_annual_expected_period")
    if expected and selected and expected.casefold() != selected.casefold():
        signals.append("selected_dataset_differs_from_expected")
    if "400 Bad Request" in error:
        signals.append("retrieval_http_400")
    if record.get("failure_stage") == "filter_resolution":
        signals.append("filter_resolution_rejected")

    expected_multivalues = _requested_multivalues(case)
    coverage = "not_assessed"
    omitted: dict[str, list[str]] = {}
    if status == "supported" and answer is not None:
        coverage = "supported_answer_emitted"
    if status == "known_unsupported" and answer is not None:
        coverage = "unsupported_answer_emitted_not_fully_assessed"
        if expected_multivalues:
            provided = defaultdict(set)
            for dimension, value in answer.get("provenance", {}).get("filters", []):
                provided[dimension].add(value)
            omitted = {
                dim: [value for value in values if value not in provided[dim]]
                for dim, values in expected_multivalues.items()
                if any(value not in provided[dim] for value in values)
            }
            if omitted:
                coverage = "unsupported_request_partially_answered"
                signals.append("requested_multi_values_omitted")

    safe_label = "not_applicable"
    if status == "safe_failure":
        if answer is not None:
            safe_label = "answer_emitted"
        elif outcome == "safe_failure":
            # A matching invalid country code is direct evidence of rejection.
            # An arbitrary error or malformed JSON is not evidence of recognition.
            if (
                "No code matches" in error
                and "ATLANTIS" in error.upper()
                and "atlantis" in case["question"].lower()
            ):
                safe_label = "reason_aligned_rejection_observed"
            else:
                safe_label = "no_answer_reason_not_established"
        else:
            safe_label = "no_answer_not_classified_safe"

    return {
        "case_id": case["case_id"],
        "outcome_original": outcome,
        "observed_failure_stage": record.get("failure_stage"),
        "diagnostic_signals": signals,
        "request_coverage": coverage,
        "requested_multi_values": expected_multivalues,
        "omitted_requested_values": omitted,
        "safe_failure_interpretation": safe_label,
        "answer_emitted": answer is not None,
        "source_and_computation_fields_present": (
            _is_structurally_present(answer) if answer is not None else None
        ),
        "provenance_violation_flag_original": record["provenance_violation"],
    }


def audit_robustness_report(report: dict[str, Any]) -> dict[str, Any]:
    """Build deterministic companion metrics from an existing Stage 11 run."""
    if report.get("schema_version") != "1.0":
        raise ValueError("Expected a Stage 11 report with schema_version '1.0'.")
    runs = report.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Stage 11 report must contain a runs list.")

    output_runs = []
    for run in runs:
        records = run["cases"]
        annotated = [annotate_case(record) for record in records]
        supported = [
            record
            for record in records
            if record["case"]["expected_capability"] == "supported"
        ]
        scored = [
            record for record in supported if record.get("benchmark_score") is not None
        ]
        score_denominators = {
            "dataset_accuracy": len(scored),
            "filters_accuracy": sum(
                score["benchmark_score"].get("filters_scored", False)
                for score in scored
            ),
            "operation_accuracy": len(scored),
            "exact_match_accuracy": len(scored),
        }
        output_runs.append(
            {
                "config": run["config"],
                "counts": {
                    "all_cases": len(records),
                    "supported_cases": len(supported),
                    "scored_supported_cases": len(scored),
                    "unsupported_request_partially_answered": sum(
                        a["request_coverage"]
                        == "unsupported_request_partially_answered"
                        for a in annotated
                    ),
                    "safe_failure_no_answer_reason_not_established": sum(
                        a["safe_failure_interpretation"]
                        == "no_answer_reason_not_established"
                        for a in annotated
                    ),
                    "safe_failure_reason_aligned_rejection_observed": sum(
                        a["safe_failure_interpretation"]
                        == "reason_aligned_rejection_observed"
                        for a in annotated
                    ),
                    "answer_objects": sum(a["answer_emitted"] for a in annotated),
                    "answer_objects_with_source_and_computation_fields": sum(
                        a["source_and_computation_fields_present"] is True
                        for a in annotated
                    ),
                    "original_provenance_violation_flags": sum(
                        a["provenance_violation_flag_original"] for a in annotated
                    ),
                },
                "score_denominators": score_denominators,
                "diagnostic_signal_counts": dict(
                    sorted(
                        Counter(
                            signal
                            for a in annotated
                            for signal in a["diagnostic_signals"]
                        ).items()
                    )
                ),
                "case_annotations": annotated,
            }
        )
    return {
        "diagnostic_schema_version": "1.0",
        "source_schema_version": report["schema_version"],
        "scope_note": (
            "Retrospective deterministic annotations of the original saved run; "
            "observed failure stages and original outcomes are unchanged. "
            "Signals are not independently proven root causes. "
            "Field-presence checks do not independently validate Eurostat values."
        ),
        "runs": output_runs,
    }


def audit_saved_report(input_path: Path, output_path: Path) -> None:
    """Write only a companion file, never overwrite original run evidence."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different files.")
    source = json.loads(input_path.read_text(encoding="utf-8"))
    audit = audit_robustness_report(source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
