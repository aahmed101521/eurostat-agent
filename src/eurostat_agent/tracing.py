from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class TraceEvent:
    """One observed stage in an execution trace."""

    stage: str
    started_at: datetime
    ended_at: datetime
    duration_seconds: float
    success: bool
    metadata: tuple[tuple[str, str], ...]
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class ExecutionTrace:
    """Ordered execution events for one agent execution."""

    execution_id: str
    events: tuple[TraceEvent, ...]


def trace_event_to_dict(event: TraceEvent) -> dict[str, object]:
    """Serialize a trace event into deterministic built-in Python values."""

    return {
        "stage": event.stage,
        "started_at": event.started_at.isoformat(),
        "ended_at": event.ended_at.isoformat(),
        "duration_seconds": event.duration_seconds,
        "success": event.success,
        "metadata": [[key, value] for key, value in event.metadata],
        "error_type": event.error_type,
        "error_message": event.error_message,
    }


def execution_trace_to_dict(trace: ExecutionTrace) -> dict[str, object]:
    """Serialize an execution trace while preserving event order."""

    return {
        "execution_id": trace.execution_id,
        "events": [trace_event_to_dict(event) for event in trace.events],
    }
