from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from eurostat_agent.tracing import (
    ExecutionTrace,
    TraceEvent,
    execution_trace_to_dict,
    trace_event_to_dict,
)


def test_trace_event_is_immutable() -> None:
    event = TraceEvent(
        stage="planning",
        started_at=datetime(2026, 9, 17, 12, 0, 0, tzinfo=UTC),
        ended_at=datetime(2026, 9, 17, 12, 0, 1, tzinfo=UTC),
        duration_seconds=1.0,
        success=True,
        metadata=(("model", "qwen3.5:latest"),),
    )

    with pytest.raises(FrozenInstanceError):
        event.stage = "retrieval"  # type: ignore[misc]


def test_trace_event_supports_failure_information() -> None:
    event = TraceEvent(
        stage="dataset_selection",
        started_at=datetime(2026, 9, 17, 12, 0, 0, tzinfo=UTC),
        ended_at=datetime(2026, 9, 17, 12, 0, 2, tzinfo=UTC),
        duration_seconds=2.0,
        success=False,
        metadata=(),
        error_type="ValueError",
        error_message="No verified dataset candidate was selected.",
    )

    assert event.success is False
    assert event.error_type == "ValueError"
    assert event.error_message == "No verified dataset candidate was selected."


def test_trace_event_serialization_is_deterministic() -> None:
    event = TraceEvent(
        stage="planner_model_call",
        started_at=datetime(2026, 9, 17, 12, 0, 0, tzinfo=UTC),
        ended_at=datetime(2026, 9, 17, 12, 0, 0, 250000, tzinfo=UTC),
        duration_seconds=0.25,
        success=True,
        metadata=(
            ("backend", "ollama"),
            ("model", "llama3.2:3b"),
            ("prompt_variant", "constrained"),
        ),
    )

    assert trace_event_to_dict(event) == {
        "stage": "planner_model_call",
        "started_at": "2026-09-17T12:00:00+00:00",
        "ended_at": "2026-09-17T12:00:00.250000+00:00",
        "duration_seconds": 0.25,
        "success": True,
        "metadata": [
            ["backend", "ollama"],
            ["model", "llama3.2:3b"],
            ["prompt_variant", "constrained"],
        ],
        "error_type": None,
        "error_message": None,
    }


def test_execution_trace_preserves_event_order() -> None:
    planning = TraceEvent(
        stage="planning",
        started_at=datetime(2026, 9, 17, 12, 0, 0, tzinfo=UTC),
        ended_at=datetime(2026, 9, 17, 12, 0, 1, tzinfo=UTC),
        duration_seconds=1.0,
        success=True,
        metadata=(),
    )
    catalogue_search = TraceEvent(
        stage="catalogue_search",
        started_at=datetime(2026, 9, 17, 12, 0, 1, tzinfo=UTC),
        ended_at=datetime(2026, 9, 17, 12, 0, 3, tzinfo=UTC),
        duration_seconds=2.0,
        success=True,
        metadata=(("candidate_count", "5"),),
    )

    trace = ExecutionTrace(
        execution_id="execution-001",
        events=(planning, catalogue_search),
    )

    assert trace.events == (planning, catalogue_search)


def test_execution_trace_serialization_is_deterministic() -> None:
    event = TraceEvent(
        stage="retrieval",
        started_at=datetime(2026, 9, 17, 12, 0, 0, tzinfo=UTC),
        ended_at=datetime(2026, 9, 17, 12, 0, 1, tzinfo=UTC),
        duration_seconds=1.0,
        success=True,
        metadata=(
            ("dataset", "DEMO_PJAN"),
            ("cache_hit", "false"),
        ),
    )
    trace = ExecutionTrace(
        execution_id="execution-001",
        events=(event,),
    )

    assert execution_trace_to_dict(trace) == {
        "execution_id": "execution-001",
        "events": [
            {
                "stage": "retrieval",
                "started_at": "2026-09-17T12:00:00+00:00",
                "ended_at": "2026-09-17T12:00:01+00:00",
                "duration_seconds": 1.0,
                "success": True,
                "metadata": [
                    ["dataset", "DEMO_PJAN"],
                    ["cache_hit", "false"],
                ],
                "error_type": None,
                "error_message": None,
            }
        ],
    }


def test_execution_trace_is_immutable() -> None:
    trace = ExecutionTrace(
        execution_id="execution-001",
        events=(),
    )

    with pytest.raises(FrozenInstanceError):
        trace.execution_id = "execution-002"  # type: ignore[misc]
