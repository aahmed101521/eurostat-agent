from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

import httpx

from eurostat_agent.benchmark import CORE_BENCHMARK_CASES
from eurostat_agent.catalogue import DatasetIndex, parse_catalogue
from eurostat_agent.catalogue_client import CATALOGUE_TOC_URL
from eurostat_agent.client import EurostatClient
from eurostat_agent.controller import (
    ControllerClient,
    DatasetSelector,
    QuestionPlanner,
)
from eurostat_agent.evaluation import (
    BenchmarkCase,
    BenchmarkFailure,
    BenchmarkResult,
    BenchmarkRun,
    BenchmarkSummary,
    run_controller_benchmark,
    summarize_benchmark_scores,
)
from eurostat_agent.experiments import (
    ExperimentComparison,
    ExperimentConfig,
    run_experiment_comparison,
    write_experiment_report,
)
from eurostat_agent.model import OpenAICompatibleModel
from eurostat_agent.planner import (
    CompletionModel,
    JsonDatasetSelector,
    JsonPlanner,
)

OLLAMA_BASE_URL = "http://localhost:11434/v1"

OLLAMA_GENERATION_OPTIONS: dict[str, object] = {
    "temperature": 0.0,
    "seed": 42,
}

CONSTRAINED_PLANNER_PROMPT_PREFIX = """\
You are participating in a controlled Eurostat planning experiment.

Return exactly one JSON object and nothing else.
Do not use markdown, code fences, explanations, notes, or commentary.

The JSON object must contain exactly these keys:

{
  "dataset_query": string,
  "filters": object,
  "operation": string
}

The planner prepares a question for a separate deterministic dataset-search
and code-resolution pipeline. It does not choose dataset codes or statistical
values itself.

Rules for "dataset_query":

- Use a concise statistical topic suitable for catalogue discovery.
- Describe both the main statistical concept and any breakdown dimensions
  that are essential to distinguish the required dataset.
- Include relevant dimensions such as age, sex, citizenship, region, or
  economic activity when the question requires that breakdown.
- Do not include observation-specific values such as a particular country,
  year, age value, sex value, unit value, or other requested category.
- Do not guess an exact Eurostat dataset title.
- Do not invent or return a Eurostat dataset code.
- Observation-specific values belong in "filters", not "dataset_query".

For example, a question asking for employment of young men in France in 2023
could use a dataset query such as "employment by age and sex", while France,
2023, the requested age, and male belong in "filters".

Rules for "filters":

- "filters" must be one JSON object, never a list.
- Use Eurostat dimension identifiers when their meaning is clear.
- Use these standard identifiers when applicable:
  - geography -> "geo"
  - year or time period -> "TIME_PERIOD"
  - age -> "age"
  - sex -> "sex"
  - frequency -> "freq"
  - unit -> "unit"
- Do not use synonyms such as "country", "year", "time", or "gender"
  as filter keys when the standard identifier above applies.
- Filter values should remain human-readable values from the question.
- Do not invent coded values such as "BE", "Y20", "F", "A", or "NR".
  Deterministic code resolution happens later.
- Preserve all material constraints needed to answer the question.

Rules for "operation":

- "operation" must be exactly one of:
  "none", "sum", "difference", "ratio", "percentage_change".
- Use "none" for a direct statistical lookup.
- Do not use operations such as "get", "retrieve", or "lookup".
- Do not calculate statistical values yourself.
"""


CONSTRAINED_SELECTOR_PROMPT_PREFIX = """\
You are participating in a controlled Eurostat dataset-selection experiment.

Return exactly one JSON object and nothing else.
Do not use markdown, code fences, explanations, notes, or commentary.

The JSON object must contain exactly this key:

{
  "dataset_code": string
}

Rules:

- Select the best dataset only from the candidate datasets supplied in the task.
- Copy the selected dataset code exactly as it appears in the candidate list.
- Do not invent a dataset code.
- Do not return filters, values, calculations, or additional keys.
- Do not use outside knowledge to introduce a dataset that is not a candidate.
"""

OLLAMA_EXPERIMENT_CONFIGS = (
    ExperimentConfig(
        name="llama32-3b-baseline",
        model="llama3.2:3b",
        prompt_variant="baseline",
    ),
    ExperimentConfig(
        name="llama32-3b-constrained",
        model="llama3.2:3b",
        prompt_variant="constrained",
    ),
    ExperimentConfig(
        name="qwen35-baseline",
        model="qwen3.5:latest",
        prompt_variant="baseline",
    ),
    ExperimentConfig(
        name="qwen35-constrained",
        model="qwen3.5:latest",
        prompt_variant="constrained",
    ),
)

ProgressReporter = Callable[[str], None]


class PrefixedCompletionModel:
    def __init__(
        self,
        model: CompletionModel,
        prefix: str,
    ) -> None:
        self._model = model
        self._prefix = prefix

    def complete(
        self,
        prompt: str,
    ) -> str:
        return self._model.complete(f"{self._prefix}\n\n{prompt}")


def select_ollama_configs(
    names: tuple[str, ...] | None = None,
) -> tuple[ExperimentConfig, ...]:
    if names is None:
        return OLLAMA_EXPERIMENT_CONFIGS

    by_name = {config.name: config for config in OLLAMA_EXPERIMENT_CONFIGS}

    unknown = tuple(name for name in names if name not in by_name)

    if unknown:
        raise ValueError("Unknown experiment configuration: " + ", ".join(unknown))

    return tuple(by_name[name] for name in names)


def select_benchmark_cases(
    indices: tuple[int, ...] | None = None,
) -> tuple[BenchmarkCase, ...]:
    if indices is None:
        return CORE_BENCHMARK_CASES

    selected: list[BenchmarkCase] = []

    for index in indices:
        if not 1 <= index <= len(CORE_BENCHMARK_CASES):
            raise ValueError(
                f"Benchmark case index must be between 1 and "
                f"{len(CORE_BENCHMARK_CASES)}."
            )

        selected.append(CORE_BENCHMARK_CASES[index - 1])

    return tuple(selected)


def build_ollama_planner_selector(
    config: ExperimentConfig,
    *,
    http_client: httpx.Client,
    base_url: str = OLLAMA_BASE_URL,
    request_options: Mapping[str, object] | None = None,
) -> tuple[QuestionPlanner, DatasetSelector]:
    model = OpenAICompatibleModel(
        base_url=base_url,
        api_key="ollama",
        model=config.model,
        http_client=http_client,
        request_options=(
            OLLAMA_GENERATION_OPTIONS if request_options is None else request_options
        ),
    )

    if config.prompt_variant == "baseline":
        planner_model: CompletionModel = model
        selector_model: CompletionModel = model

    elif config.prompt_variant == "constrained":
        planner_model = PrefixedCompletionModel(
            model,
            CONSTRAINED_PLANNER_PROMPT_PREFIX,
        )

        selector_model = PrefixedCompletionModel(
            model,
            CONSTRAINED_SELECTOR_PROMPT_PREFIX,
        )

    else:
        raise ValueError(f"Unsupported prompt variant: {config.prompt_variant}")

    return (
        JsonPlanner(planner_model),
        JsonDatasetSelector(selector_model),
    )


def fetch_live_catalogue_index(
    *,
    http_client: httpx.Client,
) -> DatasetIndex:
    response = http_client.get(CATALOGUE_TOC_URL)
    response.raise_for_status()

    return DatasetIndex(records=parse_catalogue(response.text))


def _build_combined_benchmark_run(
    *,
    cases: tuple[BenchmarkCase, ...],
    results: tuple[BenchmarkResult, ...],
    failures: tuple[BenchmarkFailure, ...],
) -> BenchmarkRun:
    total_cases = len(cases)
    completed_cases = len(results)
    failed_cases = len(failures)

    completion_rate = completed_cases / total_cases if total_cases else 0.0

    if completed_cases == 0:
        dataset_accuracy = 0.0
        filters_accuracy = 0.0
        operation_accuracy = 0.0
        exact_match_accuracy = 0.0
    else:
        score_summary = summarize_benchmark_scores(
            tuple(result.score for result in results)
        )

        dataset_accuracy = score_summary.dataset_accuracy
        filters_accuracy = score_summary.filters_accuracy
        operation_accuracy = score_summary.operation_accuracy
        exact_match_accuracy = score_summary.exact_match_accuracy

    return BenchmarkRun(
        results=results,
        failures=failures,
        summary=BenchmarkSummary(
            total_cases=total_cases,
            completed_cases=completed_cases,
            failed_cases=failed_cases,
            completion_rate=completion_rate,
            dataset_accuracy=dataset_accuracy,
            filters_accuracy=filters_accuracy,
            operation_accuracy=operation_accuracy,
            exact_match_accuracy=(exact_match_accuracy),
        ),
    )


def run_controller_benchmark_with_progress(
    cases: tuple[BenchmarkCase, ...],
    *,
    config_name: str,
    planner: QuestionPlanner,
    selector: DatasetSelector,
    client: ControllerClient,
    index: DatasetIndex,
    retrieved_at: datetime,
    progress: ProgressReporter | None = None,
) -> BenchmarkRun:
    results: list[BenchmarkResult] = []
    failures: list[BenchmarkFailure] = []

    total_cases = len(cases)

    for position, case in enumerate(
        cases,
        start=1,
    ):
        if progress is not None:
            progress(f"  [{position}/{total_cases}] {case.question}")

        single_run = run_controller_benchmark(
            (case,),
            planner=planner,
            selector=selector,
            client=client,
            index=index,
            retrieved_at=retrieved_at,
        )

        results.extend(single_run.results)
        failures.extend(single_run.failures)

        if progress is None:
            continue

        if single_run.failures:
            failure = single_run.failures[0]

            progress(f"      FAILED | {failure.error_type}: {failure.error_message}")
        else:
            score = single_run.results[0].score

            progress(
                "      done | "
                f"dataset={score.dataset_match} "
                f"filters={score.filters_match} "
                f"operation={score.operation_match} "
                f"exact={score.exact_match}"
            )

    if progress is not None:
        progress(f"  completed {config_name}")

    return _build_combined_benchmark_run(
        cases=cases,
        results=tuple(results),
        failures=tuple(failures),
    )


def run_ollama_experiment_suite(
    *,
    output_path: Path,
    configs: tuple[
        ExperimentConfig,
        ...,
    ] = OLLAMA_EXPERIMENT_CONFIGS,
    cases: tuple[
        BenchmarkCase,
        ...,
    ] = CORE_BENCHMARK_CASES,
    retrieved_at: datetime | None = None,
    model_timeout_seconds: float = 600.0,
    eurostat_timeout_seconds: float = 120.0,
    catalogue_timeout_seconds: float = 120.0,
    progress: ProgressReporter | None = None,
) -> ExperimentComparison:
    run_retrieved_at = retrieved_at if retrieved_at is not None else datetime.now(UTC)

    if progress is not None:
        progress("Loading Eurostat catalogue...")

    with httpx.Client(timeout=catalogue_timeout_seconds) as catalogue_http_client:
        index = fetch_live_catalogue_index(
            http_client=catalogue_http_client,
        )

    if progress is not None:
        progress("Catalogue ready.")

    def benchmark_executor(
        config: ExperimentConfig,
    ) -> BenchmarkRun:
        if progress is not None:
            progress("")
            progress(f"Starting {config.name}")
            progress(f"  model:  {config.model}")
            progress(f"  prompt: {config.prompt_variant}")

        with httpx.Client(timeout=model_timeout_seconds) as model_http_client:
            planner, selector = build_ollama_planner_selector(
                config,
                http_client=(model_http_client),
            )

            with httpx.Client(
                timeout=(eurostat_timeout_seconds)
            ) as eurostat_http_client:
                eurostat_client = EurostatClient(
                    http_client=(eurostat_http_client),
                )

                return run_controller_benchmark_with_progress(
                    cases,
                    config_name=config.name,
                    planner=planner,
                    selector=selector,
                    client=eurostat_client,
                    index=index,
                    retrieved_at=(run_retrieved_at),
                    progress=progress,
                )

    comparison = run_experiment_comparison(
        configs,
        benchmark_executor=benchmark_executor,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_experiment_report(
        comparison,
        output_path,
    )

    if progress is not None:
        progress("")
        progress(f"Report written to {output_path}")

    return comparison
