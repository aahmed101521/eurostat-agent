import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime


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


def _utc_now() -> datetime:
    return datetime.now(UTC)


class TraceRecorder:
    """Record execution events without changing the observed computation."""

    def __init__(
        self,
        execution_id: str,
        *,
        wall_clock: Callable[[], datetime] = _utc_now,
        monotonic_clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._execution_id = execution_id
        self._wall_clock = wall_clock
        self._monotonic_clock = monotonic_clock
        self._events: list[TraceEvent] = []

    @property
    def trace(self) -> ExecutionTrace:
        """Return an immutable snapshot of the events recorded so far."""

        return ExecutionTrace(
            execution_id=self._execution_id,
            events=tuple(self._events),
        )

    @contextmanager
    def stage(
        self,
        stage: str,
        *,
        metadata: tuple[tuple[str, str], ...] = (),
    ) -> Iterator[None]:
        """Time one execution stage and record success or failure."""

        started_at = self._wall_clock()
        started_monotonic = self._monotonic_clock()

        try:
            yield
        except Exception as exc:
            ended_at = self._wall_clock()
            duration_seconds = self._monotonic_clock() - started_monotonic
            self._events.append(
                TraceEvent(
                    stage=stage,
                    started_at=started_at,
                    ended_at=ended_at,
                    duration_seconds=duration_seconds,
                    success=False,
                    metadata=metadata,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
            raise
        else:
            ended_at = self._wall_clock()
            duration_seconds = self._monotonic_clock() - started_monotonic
            self._events.append(
                TraceEvent(
                    stage=stage,
                    started_at=started_at,
                    ended_at=ended_at,
                    duration_seconds=duration_seconds,
                    success=True,
                    metadata=metadata,
                )
            )


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
