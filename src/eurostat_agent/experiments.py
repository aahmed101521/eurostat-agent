import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from eurostat_agent.evaluation import BenchmarkRun, benchmark_summary_to_dict
from eurostat_agent.judge import JudgeRun, JudgeRunSummary


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model: str
    prompt_variant: str


@dataclass(frozen=True)
class ExperimentResult:
    config: ExperimentConfig
    benchmark_run: BenchmarkRun
    judge_run: JudgeRun | None


@dataclass(frozen=True)
class ExperimentFailure:
    config: ExperimentConfig
    stage: Literal["benchmark", "judge"]
    error_type: str
    error_message: str


@dataclass(frozen=True)
class ExperimentComparison:
    results: tuple[ExperimentResult, ...]
    failures: tuple[ExperimentFailure, ...]


BenchmarkExecutor = Callable[[ExperimentConfig], BenchmarkRun]
JudgeExecutor = Callable[[ExperimentConfig, BenchmarkRun], JudgeRun]


def experiment_config_to_dict(
    config: ExperimentConfig,
) -> dict[str, str]:
    return {
        "name": config.name,
        "model": config.model,
        "prompt_variant": config.prompt_variant,
    }


def judge_run_summary_to_dict(
    summary: JudgeRunSummary,
) -> dict[str, int | float]:
    return {
        "total_cases": summary.total_cases,
        "completed_cases": summary.completed_cases,
        "failed_cases": summary.failed_cases,
        "completion_rate": summary.completion_rate,
        "pass_rate": summary.pass_rate,
        "mean_relevance": summary.mean_relevance,
        "mean_factual_fidelity": summary.mean_factual_fidelity,
        "mean_provenance_fidelity": summary.mean_provenance_fidelity,
        "mean_clarity": summary.mean_clarity,
    }


def experiment_result_to_dict(
    result: ExperimentResult,
) -> dict[str, object]:
    judge_summary: dict[str, int | float] | None = None

    if result.judge_run is not None:
        judge_summary = judge_run_summary_to_dict(result.judge_run.summary)

    return {
        "name": result.config.name,
        "model": result.config.model,
        "prompt_variant": result.config.prompt_variant,
        "benchmark": benchmark_summary_to_dict(result.benchmark_run.summary),
        "judge": judge_summary,
    }


def experiment_failure_to_dict(
    failure: ExperimentFailure,
) -> dict[str, object]:
    return {
        "config": experiment_config_to_dict(failure.config),
        "stage": failure.stage,
        "error_type": failure.error_type,
        "error_message": failure.error_message,
    }


def experiment_comparison_to_dict(
    comparison: ExperimentComparison,
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "experiments": [
            experiment_result_to_dict(result) for result in comparison.results
        ],
        "failures": [
            experiment_failure_to_dict(failure) for failure in comparison.failures
        ],
    }


def write_experiment_report(
    comparison: ExperimentComparison,
    path: Path,
) -> None:
    path.write_text(
        json.dumps(
            experiment_comparison_to_dict(comparison),
            indent=2,
        ),
        encoding="utf-8",
    )


def run_experiment(
    config: ExperimentConfig,
    *,
    benchmark_executor: BenchmarkExecutor,
    judge_executor: JudgeExecutor | None = None,
) -> ExperimentResult:
    benchmark_run = benchmark_executor(config)

    judge_run = (
        judge_executor(config, benchmark_run) if judge_executor is not None else None
    )

    return ExperimentResult(
        config=config,
        benchmark_run=benchmark_run,
        judge_run=judge_run,
    )


def run_experiment_comparison(
    configs: tuple[ExperimentConfig, ...],
    *,
    benchmark_executor: BenchmarkExecutor,
    judge_executor: JudgeExecutor | None = None,
) -> ExperimentComparison:
    results: list[ExperimentResult] = []
    failures: list[ExperimentFailure] = []

    for config in configs:
        try:
            benchmark_run = benchmark_executor(config)
        except Exception as exc:
            failures.append(
                ExperimentFailure(
                    config=config,
                    stage="benchmark",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
            continue

        if judge_executor is None:
            results.append(
                ExperimentResult(
                    config=config,
                    benchmark_run=benchmark_run,
                    judge_run=None,
                )
            )
            continue

        try:
            judge_run = judge_executor(
                config,
                benchmark_run,
            )
        except Exception as exc:
            results.append(
                ExperimentResult(
                    config=config,
                    benchmark_run=benchmark_run,
                    judge_run=None,
                )
            )
            failures.append(
                ExperimentFailure(
                    config=config,
                    stage="judge",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
            continue

        results.append(
            ExperimentResult(
                config=config,
                benchmark_run=benchmark_run,
                judge_run=judge_run,
            )
        )

    return ExperimentComparison(
        results=tuple(results),
        failures=tuple(failures),
    )
