from collections.abc import Callable
from dataclasses import dataclass

from eurostat_agent.provenance import DeterministicAnswer


@dataclass(frozen=True)
class BenchmarkCase:
    question: str
    expected_dataset_code: str
    expected_filters: tuple[tuple[str, str], ...]
    expected_operation: str
    score_filters: bool = True


@dataclass(frozen=True)
class BenchmarkScore:
    dataset_match: bool
    filters_match: bool
    operation_match: bool
    exact_match: bool
    filters_scored: bool = True


@dataclass(frozen=True)
class BenchmarkResult:
    case: BenchmarkCase
    answer: DeterministicAnswer
    score: BenchmarkScore


@dataclass(frozen=True)
class BenchmarkFailure:
    case: BenchmarkCase
    error_type: str
    error_message: str


@dataclass(frozen=True)
class BenchmarkSummary:
    total_cases: int
    completed_cases: int
    failed_cases: int
    completion_rate: float
    dataset_accuracy: float
    filters_accuracy: float
    operation_accuracy: float
    exact_match_accuracy: float


@dataclass(frozen=True)
class BenchmarkRun:
    results: tuple[BenchmarkResult, ...]
    failures: tuple[BenchmarkFailure, ...]
    summary: BenchmarkSummary


def score_benchmark_case(
    case: BenchmarkCase,
    *,
    actual_dataset_code: str,
    actual_filters: tuple[tuple[str, str], ...],
    actual_operation: str,
) -> BenchmarkScore:
    dataset_match = actual_dataset_code == case.expected_dataset_code

    filters_match = (
        True if not case.score_filters else actual_filters == case.expected_filters
    )

    operation_match = actual_operation == case.expected_operation

    return BenchmarkScore(
        dataset_match=dataset_match,
        filters_match=filters_match,
        operation_match=operation_match,
        exact_match=(dataset_match and filters_match and operation_match),
        filters_scored=case.score_filters,
    )


def score_deterministic_answer(
    case: BenchmarkCase,
    answer: DeterministicAnswer,
) -> BenchmarkScore:
    return score_benchmark_case(
        case,
        actual_dataset_code=answer.provenance.dataset_code,
        actual_filters=answer.provenance.filters,
        actual_operation=answer.computation.operation,
    )


def build_benchmark_result(
    case: BenchmarkCase,
    answer: DeterministicAnswer,
) -> BenchmarkResult:
    return BenchmarkResult(
        case=case,
        answer=answer,
        score=score_deterministic_answer(
            case,
            answer,
        ),
    )


def summarize_benchmark_scores(
    scores: tuple[BenchmarkScore, ...],
) -> BenchmarkSummary:
    total_cases = len(scores)

    if total_cases == 0:
        raise ValueError("Cannot summarize an empty benchmark.")

    scored_filters = tuple(score for score in scores if score.filters_scored)

    filters_accuracy = (
        sum(score.filters_match for score in scored_filters) / len(scored_filters)
        if scored_filters
        else 0.0
    )

    return BenchmarkSummary(
        total_cases=total_cases,
        completed_cases=total_cases,
        failed_cases=0,
        completion_rate=1.0,
        dataset_accuracy=(sum(score.dataset_match for score in scores) / total_cases),
        filters_accuracy=filters_accuracy,
        operation_accuracy=(
            sum(score.operation_match for score in scores) / total_cases
        ),
        exact_match_accuracy=(sum(score.exact_match for score in scores) / total_cases),
    )


def run_benchmark(
    cases: tuple[BenchmarkCase, ...],
    answer_question: Callable[[str], DeterministicAnswer],
) -> BenchmarkRun:
    results: list[BenchmarkResult] = []
    failures: list[BenchmarkFailure] = []

    for case in cases:
        try:
            answer = answer_question(case.question)
        except Exception as exc:
            failures.append(
                BenchmarkFailure(
                    case=case,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
            continue

        results.append(
            build_benchmark_result(
                case,
                answer,
            )
        )

    completed_cases = len(results)
    failed_cases = len(failures)
    total_cases = len(cases)

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

    summary = BenchmarkSummary(
        total_cases=total_cases,
        completed_cases=completed_cases,
        failed_cases=failed_cases,
        completion_rate=completion_rate,
        dataset_accuracy=dataset_accuracy,
        filters_accuracy=filters_accuracy,
        operation_accuracy=operation_accuracy,
        exact_match_accuracy=exact_match_accuracy,
    )

    return BenchmarkRun(
        results=tuple(results),
        failures=tuple(failures),
        summary=summary,
    )
