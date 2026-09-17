"""Deterministic operational reporting for execution traces."""

from dataclasses import dataclass

from eurostat_agent.tracing import ExecutionTrace


@dataclass(frozen=True)
class ModelUsage:
    """Observed model metadata for one traced execution stage."""

    stage: str
    model: str
    backend: str | None
    prompt_variant: str | None
    duration_seconds: float


@dataclass(frozen=True)
class TraceFailure:
    """One unsuccessful traced stage."""

    stage: str
    error_type: str | None
    error_message: str | None


@dataclass(frozen=True)
class OperationalReport:
    """Compact operational summary derived only from observed trace events."""

    execution_id: str
    success: bool
    total_duration_seconds: float | None
    stage_durations: tuple[tuple[str, float], ...]
    model_call_count: int
    model_usage: tuple[ModelUsage, ...]
    cache_hits: int
    cache_misses: int
    failures: tuple[TraceFailure, ...]


def build_operational_report(trace: ExecutionTrace) -> OperationalReport:
    """Summarize one execution trace without inventing missing telemetry."""

    total_duration_seconds = next(
        (
            event.duration_seconds
            for event in reversed(trace.events)
            if event.stage == "controller_total"
        ),
        None,
    )

    model_usage: list[ModelUsage] = []
    cache_hits = 0
    cache_misses = 0
    failures: list[TraceFailure] = []

    for event in trace.events:
        metadata = dict(event.metadata)

        model = metadata.get("model")
        if model is not None:
            model_usage.append(
                ModelUsage(
                    stage=event.stage,
                    model=model,
                    backend=metadata.get("backend"),
                    prompt_variant=metadata.get("prompt_variant"),
                    duration_seconds=event.duration_seconds,
                )
            )

        if event.stage == "metadata_cache":
            cache_hit = metadata.get("cache_hit")
            if cache_hit == "true":
                cache_hits += 1
            elif cache_hit == "false":
                cache_misses += 1

        if not event.success:
            failures.append(
                TraceFailure(
                    stage=event.stage,
                    error_type=event.error_type,
                    error_message=event.error_message,
                )
            )

    return OperationalReport(
        execution_id=trace.execution_id,
        success=all(event.success for event in trace.events),
        total_duration_seconds=total_duration_seconds,
        stage_durations=tuple(
            (event.stage, event.duration_seconds) for event in trace.events
        ),
        model_call_count=len(model_usage),
        model_usage=tuple(model_usage),
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        failures=tuple(failures),
    )


def operational_report_to_dict(report: OperationalReport) -> dict[str, object]:
    """Serialize an operational report into deterministic built-in values."""

    return {
        "execution_id": report.execution_id,
        "success": report.success,
        "total_duration_seconds": report.total_duration_seconds,
        "stage_durations": [
            [stage, duration] for stage, duration in report.stage_durations
        ],
        "model_call_count": report.model_call_count,
        "model_usage": [
            {
                "stage": usage.stage,
                "model": usage.model,
                "backend": usage.backend,
                "prompt_variant": usage.prompt_variant,
                "duration_seconds": usage.duration_seconds,
            }
            for usage in report.model_usage
        ],
        "cache_hits": report.cache_hits,
        "cache_misses": report.cache_misses,
        "failures": [
            {
                "stage": failure.stage,
                "error_type": failure.error_type,
                "error_message": failure.error_message,
            }
            for failure in report.failures
        ],
    }
