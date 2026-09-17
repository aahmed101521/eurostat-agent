import json
from pathlib import Path

import pytest

from eurostat_agent.evaluation import BenchmarkRun, BenchmarkSummary
from eurostat_agent.experiments import (
    ExperimentComparison,
    ExperimentConfig,
    ExperimentFailure,
    ExperimentResult,
    experiment_comparison_to_dict,
    run_experiment,
    run_experiment_comparison,
    write_experiment_report,
)
from eurostat_agent.judge import JudgeRun, JudgeRunSummary


def _benchmark_run(
    *,
    completion_rate: float = 1.0,
    exact_match_accuracy: float = 0.75,
) -> BenchmarkRun:
    return BenchmarkRun(
        results=(),
        failures=(),
        summary=BenchmarkSummary(
            total_cases=4,
            completed_cases=4,
            failed_cases=0,
            completion_rate=completion_rate,
            dataset_accuracy=1.0,
            filters_accuracy=0.75,
            operation_accuracy=1.0,
            exact_match_accuracy=exact_match_accuracy,
        ),
    )


def _judge_run(
    *,
    pass_rate: float = 0.5,
) -> JudgeRun:
    return JudgeRun(
        results=(),
        failures=(),
        summary=JudgeRunSummary(
            total_cases=4,
            completed_cases=4,
            failed_cases=0,
            completion_rate=1.0,
            pass_rate=pass_rate,
            mean_relevance=2.0,
            mean_factual_fidelity=1.75,
            mean_provenance_fidelity=1.5,
            mean_clarity=2.0,
        ),
    )


def _config(
    name: str = "baseline",
) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        model=f"{name}-model",
        prompt_variant=f"{name}-prompt",
    )


def test_experiment_config_records_model_and_prompt_variant() -> None:
    config = ExperimentConfig(
        name="baseline-gpt",
        model="example-model",
        prompt_variant="baseline",
    )

    assert config.name == "baseline-gpt"
    assert config.model == "example-model"
    assert config.prompt_variant == "baseline"


def test_run_experiment_records_benchmark_without_judge() -> None:
    config = _config()
    benchmark = _benchmark_run()

    result = run_experiment(
        config,
        benchmark_executor=lambda received: benchmark,
    )

    assert result.config == config
    assert result.benchmark_run == benchmark
    assert result.judge_run is None


def test_run_experiment_passes_benchmark_run_to_judge() -> None:
    config = _config()
    benchmark = _benchmark_run()
    judge = _judge_run()
    seen: list[tuple[ExperimentConfig, BenchmarkRun]] = []

    def judge_executor(
        received_config: ExperimentConfig,
        received_benchmark: BenchmarkRun,
    ) -> JudgeRun:
        seen.append(
            (
                received_config,
                received_benchmark,
            )
        )
        return judge

    result = run_experiment(
        config,
        benchmark_executor=lambda received: benchmark,
        judge_executor=judge_executor,
    )

    assert seen == [(config, benchmark)]
    assert result.judge_run == judge


def test_run_experiment_propagates_executor_errors() -> None:
    config = _config()

    def benchmark_executor(
        received: ExperimentConfig,
    ) -> BenchmarkRun:
        raise RuntimeError("benchmark unavailable")

    with pytest.raises(
        RuntimeError,
        match="benchmark unavailable",
    ):
        run_experiment(
            config,
            benchmark_executor=benchmark_executor,
        )


def test_comparison_runs_multiple_configs_in_order() -> None:
    first = _config("first")
    second = _config("second")
    benchmark = _benchmark_run()
    judge = _judge_run()

    comparison = run_experiment_comparison(
        (first, second),
        benchmark_executor=lambda config: benchmark,
        judge_executor=lambda config, run: judge,
    )

    assert tuple(result.config for result in comparison.results) == (
        first,
        second,
    )

    assert tuple(result.benchmark_run for result in comparison.results) == (
        benchmark,
        benchmark,
    )

    assert tuple(result.judge_run for result in comparison.results) == (
        judge,
        judge,
    )

    assert comparison.failures == ()


def test_comparison_isolates_benchmark_failures() -> None:
    good = _config("good")
    broken = _config("broken")
    benchmark = _benchmark_run()

    def benchmark_executor(
        config: ExperimentConfig,
    ) -> BenchmarkRun:
        if config == broken:
            raise ValueError("bad planner configuration")

        return benchmark

    comparison = run_experiment_comparison(
        (good, broken),
        benchmark_executor=benchmark_executor,
    )

    assert tuple(result.config for result in comparison.results) == (good,)

    assert comparison.failures == (
        ExperimentFailure(
            config=broken,
            stage="benchmark",
            error_type="ValueError",
            error_message="bad planner configuration",
        ),
    )


def test_comparison_preserves_benchmark_when_judge_fails() -> None:
    config = _config()
    benchmark = _benchmark_run()

    def judge_executor(
        received_config: ExperimentConfig,
        received_benchmark: BenchmarkRun,
    ) -> JudgeRun:
        raise RuntimeError("judge unavailable")

    comparison = run_experiment_comparison(
        (config,),
        benchmark_executor=lambda received: benchmark,
        judge_executor=judge_executor,
    )

    assert comparison.results == (
        ExperimentResult(
            config=config,
            benchmark_run=benchmark,
            judge_run=None,
        ),
    )

    assert comparison.failures == (
        ExperimentFailure(
            config=config,
            stage="judge",
            error_type="RuntimeError",
            error_message="judge unavailable",
        ),
    )


def test_experiment_report_exposes_comparable_metrics_without_ranking() -> None:
    config = _config()

    comparison = ExperimentComparison(
        results=(
            ExperimentResult(
                config=config,
                benchmark_run=_benchmark_run(),
                judge_run=_judge_run(),
            ),
        ),
        failures=(),
    )

    payload = experiment_comparison_to_dict(comparison)

    experiments = payload["experiments"]
    assert isinstance(experiments, list)

    experiment = experiments[0]
    assert isinstance(experiment, dict)

    assert payload["schema_version"] == "1.0"

    assert experiment == {
        "name": "baseline",
        "model": "baseline-model",
        "prompt_variant": "baseline-prompt",
        "benchmark": {
            "total_cases": 4,
            "completed_cases": 4,
            "failed_cases": 0,
            "completion_rate": 1.0,
            "dataset_accuracy": 1.0,
            "filters_accuracy": 0.75,
            "operation_accuracy": 1.0,
            "exact_match_accuracy": 0.75,
        },
        "benchmark_run": {
            "schema_version": "1.0",
            "summary": {
                "total_cases": 4,
                "completed_cases": 4,
                "failed_cases": 0,
                "completion_rate": 1.0,
                "dataset_accuracy": 1.0,
                "filters_accuracy": 0.75,
                "operation_accuracy": 1.0,
                "exact_match_accuracy": 0.75,
            },
            "results": [],
            "failures": [],
        },
        "judge": {
            "total_cases": 4,
            "completed_cases": 4,
            "failed_cases": 0,
            "completion_rate": 1.0,
            "pass_rate": 0.5,
            "mean_relevance": 2.0,
            "mean_factual_fidelity": 1.75,
            "mean_provenance_fidelity": 1.5,
            "mean_clarity": 2.0,
        },
    }

    assert "best" not in payload
    assert "ranking" not in payload


def test_experiment_report_uses_null_when_judge_is_not_run() -> None:
    comparison = ExperimentComparison(
        results=(
            ExperimentResult(
                config=_config(),
                benchmark_run=_benchmark_run(),
                judge_run=None,
            ),
        ),
        failures=(),
    )

    payload = experiment_comparison_to_dict(comparison)

    experiments = payload["experiments"]
    assert isinstance(experiments, list)

    experiment = experiments[0]
    assert isinstance(experiment, dict)

    assert experiment["judge"] is None


def test_write_experiment_report_persists_json(
    tmp_path: Path,
) -> None:
    comparison = ExperimentComparison(
        results=(
            ExperimentResult(
                config=_config(),
                benchmark_run=_benchmark_run(),
                judge_run=_judge_run(),
            ),
        ),
        failures=(),
    )

    path = tmp_path / "experiment-report.json"

    write_experiment_report(
        comparison,
        path,
    )

    payload = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )

    assert payload == experiment_comparison_to_dict(comparison)
