"""Live Ollama + Eurostat execution for the Stage 11 robustness benchmark."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx

from eurostat_agent.benchmark_cases import ROBUSTNESS_BENCHMARK_CASES
from eurostat_agent.client import EurostatClient
from eurostat_agent.experiments import ExperimentConfig
from eurostat_agent.ollama_experiments import (
    build_ollama_planner_selector,
    fetch_live_catalogue_index,
    select_ollama_configs,
)
from eurostat_agent.robustness import (
    RobustnessCase,
    RobustnessComparison,
    RobustnessConfigFailure,
    RobustnessEvidence,
    RobustnessRun,
    run_robustness_case,
    summarize_robustness_evidence,
    write_robustness_report,
)

ProgressReporter = Callable[[str], None]

DEFAULT_ROBUSTNESS_CONFIG_NAMES = (
    "llama32-3b-constrained",
    "qwen35-constrained",
)


def default_robustness_configs() -> tuple[ExperimentConfig, ...]:
    """Use only constrained Stage 9 configurations for the primary Stage 11 run."""

    return select_ollama_configs(DEFAULT_ROBUSTNESS_CONFIG_NAMES)


def run_ollama_robustness_suite(
    *,
    output_path: Path,
    configs: tuple[ExperimentConfig, ...] | None = None,
    cases: tuple[RobustnessCase, ...] = ROBUSTNESS_BENCHMARK_CASES,
    retrieved_at: datetime | None = None,
    model_timeout_seconds: float = 600.0,
    eurostat_timeout_seconds: float = 120.0,
    catalogue_timeout_seconds: float = 120.0,
    progress: ProgressReporter | None = None,
) -> RobustnessComparison:
    """Run Stage 11 live while keeping normal tests independent of network/Ollama."""

    selected_configs = configs if configs is not None else default_robustness_configs()
    if progress is not None:
        progress("Loading Eurostat catalogue...")

    with httpx.Client(timeout=catalogue_timeout_seconds) as catalogue_http_client:
        index = fetch_live_catalogue_index(http_client=catalogue_http_client)

    if progress is not None:
        progress("Catalogue ready.")

    runs: list[RobustnessRun] = []
    failures: list[RobustnessConfigFailure] = []

    for config in selected_configs:
        if progress is not None:
            progress("")
            progress(f"Starting {config.name}")
            progress(f"  model:  {config.model}")
            progress(f"  prompt: {config.prompt_variant}")

        try:
            with httpx.Client(timeout=model_timeout_seconds) as model_http_client:
                planner, selector = build_ollama_planner_selector(
                    config,
                    http_client=model_http_client,
                )

                with httpx.Client(
                    timeout=eurostat_timeout_seconds
                ) as eurostat_http_client:
                    eurostat_client = EurostatClient(http_client=eurostat_http_client)
                    evidence: list[RobustnessEvidence] = []
                    total_cases = len(cases)

                    for position, case in enumerate(cases, start=1):
                        if progress is not None:
                            progress(
                                f"  [{position}/{total_cases}] "
                                f"{case.case_id} {case.question}"
                            )

                        case_retrieved_at = (
                            retrieved_at
                            if retrieved_at is not None
                            else datetime.now(UTC)
                        )
                        item = run_robustness_case(
                            case,
                            config=config,
                            planner=planner,
                            selector=selector,
                            client=eurostat_client,
                            index=index,
                            retrieved_at=case_retrieved_at,
                        )
                        evidence.append(item)

                        if progress is not None:
                            detail = item.failure_stage or "completed"
                            progress(f"      {item.outcome} | {detail}")

                    evidence_tuple = tuple(evidence)
                    runs.append(
                        RobustnessRun(
                            config=config,
                            evidence=evidence_tuple,
                            summary=summarize_robustness_evidence(evidence_tuple),
                        )
                    )
        except Exception as exc:
            failures.append(
                RobustnessConfigFailure(
                    config=config,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )

            if progress is not None:
                progress(f"  CONFIG FAILED | {type(exc).__name__}: {str(exc)}")

    comparison = RobustnessComparison(
        runs=tuple(runs),
        failures=tuple(failures),
    )
    write_robustness_report(comparison, output_path)

    if progress is not None:
        progress("")
        progress(f"Report written to {output_path}")

    return comparison
