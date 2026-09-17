from datetime import UTC, datetime

from eurostat_agent.reporting import (
    build_operational_report,
    operational_report_to_dict,
)
from eurostat_agent.tracing import ExecutionTrace, TraceEvent


def _event(
    stage: str,
    duration: float,
    *,
    success: bool = True,
    metadata: tuple[tuple[str, str], ...] = (),
    error_type: str | None = None,
    error_message: str | None = None,
) -> TraceEvent:
    return TraceEvent(
        stage=stage,
        started_at=datetime(2026, 9, 17, 18, 0, tzinfo=UTC),
        ended_at=datetime(2026, 9, 17, 18, 0, 1, tzinfo=UTC),
        duration_seconds=duration,
        success=success,
        metadata=metadata,
        error_type=error_type,
        error_message=error_message,
    )


def test_operational_report_summarizes_observed_execution_facts() -> None:
    trace = ExecutionTrace(
        execution_id="execution-001",
        events=(
            _event(
                "planning",
                0.4,
                metadata=(
                    ("backend", "ollama"),
                    ("model", "qwen3.5:latest"),
                    ("prompt_variant", "constrained"),
                ),
            ),
            _event(
                "metadata_cache",
                0.2,
                metadata=(
                    ("resource", "data_structure"),
                    ("cache_key", "demo_pjan"),
                    ("cache_hit", "false"),
                ),
            ),
            _event(
                "metadata_cache",
                0.001,
                metadata=(
                    ("resource", "data_structure"),
                    ("cache_key", "demo_pjan"),
                    ("cache_hit", "true"),
                ),
            ),
            _event(
                "dataset_selection",
                0.3,
                metadata=(
                    ("backend", "ollama"),
                    ("model", "qwen3.5:latest"),
                ),
            ),
            _event("controller_total", 1.5),
        ),
    )

    report = build_operational_report(trace)

    assert report.execution_id == "execution-001"
    assert report.success is True
    assert report.total_duration_seconds == 1.5
    assert report.model_call_count == 2
    assert report.cache_hits == 1
    assert report.cache_misses == 1
    assert len(report.model_usage) == 2
    assert report.model_usage[0].model == "qwen3.5:latest"
    assert report.model_usage[0].prompt_variant == "constrained"
    assert report.failures == ()


def test_operational_report_preserves_failure_information() -> None:
    trace = ExecutionTrace(
        execution_id="execution-failure",
        events=(
            _event(
                "planning",
                0.1,
                success=False,
                error_type="ValueError",
                error_message="planner failed",
            ),
            _event(
                "controller_total",
                0.1,
                success=False,
                error_type="ValueError",
                error_message="planner failed",
            ),
        ),
    )

    report = build_operational_report(trace)

    assert report.success is False
    assert report.total_duration_seconds == 0.1
    assert report.model_call_count == 0
    assert report.cache_hits == 0
    assert report.cache_misses == 0
    assert tuple(failure.stage for failure in report.failures) == (
        "planning",
        "controller_total",
    )


def test_operational_report_serialization_is_deterministic() -> None:
    trace = ExecutionTrace(
        execution_id="execution-001",
        events=(
            _event("retrieval", 0.25),
            _event("controller_total", 0.5),
        ),
    )

    report = build_operational_report(trace)

    assert operational_report_to_dict(report) == {
        "execution_id": "execution-001",
        "success": True,
        "total_duration_seconds": 0.5,
        "stage_durations": [
            ["retrieval", 0.25],
            ["controller_total", 0.5],
        ],
        "model_call_count": 0,
        "model_usage": [],
        "cache_hits": 0,
        "cache_misses": 0,
        "failures": [],
    }
