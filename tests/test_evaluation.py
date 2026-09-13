from datetime import UTC, datetime

import pytest

from eurostat_agent.benchmark import CORE_BENCHMARK_CASES
from eurostat_agent.evaluation import (
    BenchmarkCase,
    BenchmarkFailure,
    BenchmarkResult,
    BenchmarkScore,
    build_benchmark_result,
    run_benchmark,
    score_benchmark_case,
    score_deterministic_answer,
    summarize_benchmark_scores,
)
from eurostat_agent.provenance import (
    ComputationProvenance,
    DeterministicAnswer,
    RetrievalProvenance,
)


def test_benchmark_case_preserves_expected_semantics() -> None:
    case = BenchmarkCase(
        question="What was the population of women in Belgium in 2024?",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(
            ("TIME_PERIOD", "2024"),
            ("geo", "BE"),
            ("sex", "F"),
        ),
        expected_operation="none",
    )

    assert case.question == ("What was the population of women in Belgium in 2024?")
    assert case.expected_dataset_code == "DEMO_PJAN"
    assert case.expected_filters == (
        ("TIME_PERIOD", "2024"),
        ("geo", "BE"),
        ("sex", "F"),
    )
    assert case.expected_operation == "none"


def test_score_benchmark_case_reports_exact_match() -> None:
    case = BenchmarkCase(
        question="What was the population of women in Belgium in 2024?",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(
            ("TIME_PERIOD", "2024"),
            ("geo", "BE"),
            ("sex", "F"),
        ),
        expected_operation="none",
    )

    result = score_benchmark_case(
        case,
        actual_dataset_code="DEMO_PJAN",
        actual_filters=(
            ("TIME_PERIOD", "2024"),
            ("geo", "BE"),
            ("sex", "F"),
        ),
        actual_operation="none",
    )

    assert result.dataset_match is True
    assert result.filters_match is True
    assert result.operation_match is True
    assert result.exact_match is True


def test_score_benchmark_case_reports_partial_mismatch() -> None:
    case = BenchmarkCase(
        question="What was the population of women in Belgium in 2024?",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(
            ("TIME_PERIOD", "2024"),
            ("geo", "BE"),
            ("sex", "F"),
        ),
        expected_operation="none",
    )

    result = score_benchmark_case(
        case,
        actual_dataset_code="DEMO_PJAN",
        actual_filters=(
            ("TIME_PERIOD", "2024"),
            ("geo", "DE"),
            ("sex", "F"),
        ),
        actual_operation="none",
    )

    assert result.dataset_match is True
    assert result.filters_match is False
    assert result.operation_match is True
    assert result.exact_match is False


def test_summarize_benchmark_scores_reports_component_accuracy() -> None:
    scores = (
        BenchmarkScore(
            dataset_match=True,
            filters_match=True,
            operation_match=True,
            exact_match=True,
        ),
        BenchmarkScore(
            dataset_match=True,
            filters_match=False,
            operation_match=True,
            exact_match=False,
        ),
    )

    summary = summarize_benchmark_scores(scores)

    assert summary.total_cases == 2
    assert summary.dataset_accuracy == 1.0
    assert summary.filters_accuracy == 0.5
    assert summary.operation_accuracy == 1.0
    assert summary.exact_match_accuracy == 0.5


def test_summarize_benchmark_scores_rejects_empty_scores() -> None:
    with pytest.raises(
        ValueError,
        match="Cannot summarize an empty benchmark.",
    ):
        summarize_benchmark_scores(())


def test_score_deterministic_answer_uses_answer_provenance() -> None:
    case = BenchmarkCase(
        question="What was the population of women in Belgium in 2024?",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(
            ("TIME_PERIOD", "2024"),
            ("geo", "BE"),
            ("sex", "F"),
        ),
        expected_operation="none",
    )

    answer = DeterministicAnswer(
        value=123.0,
        unit="NR",
        computation=ComputationProvenance(
            operation="none",
            input_values=(123.0,),
            input_time_periods=("2024",),
            input_dimensions=(
                (
                    ("geo", "BE"),
                    ("sex", "F"),
                    ("unit", "NR"),
                ),
            ),
            output_value=123.0,
        ),
        provenance=RetrievalProvenance(
            dataset_code="DEMO_PJAN",
            dataset_agency="ESTAT",
            dataset_version="1.0",
            filters=(
                ("TIME_PERIOD", "2024"),
                ("geo", "BE"),
                ("sex", "F"),
            ),
            retrieved_at=datetime(
                2026,
                9,
                13,
                12,
                0,
                tzinfo=UTC,
            ),
            data_updated_at="2026-08-14T23:00:00+0200",
            source="Eurostat",
            source_url="https://ec.europa.eu/eurostat/",
        ),
    )

    result = score_deterministic_answer(
        case,
        answer,
    )

    assert result.dataset_match is True
    assert result.filters_match is True
    assert result.operation_match is True
    assert result.exact_match is True


def test_benchmark_result_preserves_case_answer_and_score() -> None:
    case = BenchmarkCase(
        question="What was the population of women in Belgium in 2024?",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(
            ("TIME_PERIOD", "2024"),
            ("geo", "BE"),
            ("sex", "F"),
        ),
        expected_operation="none",
    )

    answer = DeterministicAnswer(
        value=123.0,
        unit="NR",
        computation=ComputationProvenance(
            operation="none",
            input_values=(123.0,),
            input_time_periods=("2024",),
            input_dimensions=(
                (
                    ("geo", "BE"),
                    ("sex", "F"),
                    ("unit", "NR"),
                ),
            ),
            output_value=123.0,
        ),
        provenance=RetrievalProvenance(
            dataset_code="DEMO_PJAN",
            dataset_agency="ESTAT",
            dataset_version="1.0",
            filters=(
                ("TIME_PERIOD", "2024"),
                ("geo", "BE"),
                ("sex", "F"),
            ),
            retrieved_at=datetime(
                2026,
                9,
                13,
                12,
                0,
                tzinfo=UTC,
            ),
            data_updated_at="2026-08-14T23:00:00+0200",
            source="Eurostat",
            source_url="https://ec.europa.eu/eurostat/",
        ),
    )

    score = score_deterministic_answer(
        case,
        answer,
    )

    result = BenchmarkResult(
        case=case,
        answer=answer,
        score=score,
    )

    assert result.case == case
    assert result.answer == answer
    assert result.score == score
    assert result.score.exact_match is True


def test_build_benchmark_result_scores_answer() -> None:
    case = BenchmarkCase(
        question="What was the population of women in Belgium in 2024?",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(
            ("TIME_PERIOD", "2024"),
            ("geo", "BE"),
            ("sex", "F"),
        ),
        expected_operation="none",
    )

    answer = DeterministicAnswer(
        value=123.0,
        unit="NR",
        computation=ComputationProvenance(
            operation="none",
            input_values=(123.0,),
            input_time_periods=("2024",),
            input_dimensions=(
                (
                    ("geo", "BE"),
                    ("sex", "F"),
                    ("unit", "NR"),
                ),
            ),
            output_value=123.0,
        ),
        provenance=RetrievalProvenance(
            dataset_code="DEMO_PJAN",
            dataset_agency="ESTAT",
            dataset_version="1.0",
            filters=(
                ("TIME_PERIOD", "2024"),
                ("geo", "BE"),
                ("sex", "F"),
            ),
            retrieved_at=datetime(
                2026,
                9,
                13,
                12,
                0,
                tzinfo=UTC,
            ),
            data_updated_at="2026-08-14T23:00:00+0200",
            source="Eurostat",
            source_url="https://ec.europa.eu/eurostat/",
        ),
    )

    result = build_benchmark_result(
        case,
        answer,
    )

    assert result.case == case
    assert result.answer == answer
    assert result.score.dataset_match is True
    assert result.score.filters_match is True
    assert result.score.operation_match is True
    assert result.score.exact_match is True


def test_run_benchmark_returns_results_and_summary() -> None:
    cases = (
        BenchmarkCase(
            question="Question one",
            expected_dataset_code="DEMO_PJAN",
            expected_filters=(("TIME_PERIOD", "2024"),),
            expected_operation="none",
        ),
        BenchmarkCase(
            question="Question two",
            expected_dataset_code="DEMO_PJAN",
            expected_filters=(("TIME_PERIOD", "2023"),),
            expected_operation="none",
        ),
    )

    answers = {
        "Question one": DeterministicAnswer(
            value=100.0,
            unit="NR",
            computation=ComputationProvenance(
                operation="none",
                input_values=(100.0,),
                input_time_periods=("2024",),
                input_dimensions=(),
                output_value=100.0,
            ),
            provenance=RetrievalProvenance(
                dataset_code="DEMO_PJAN",
                dataset_agency="ESTAT",
                dataset_version="1.0",
                filters=(("TIME_PERIOD", "2024"),),
                retrieved_at=datetime(
                    2026,
                    9,
                    13,
                    12,
                    0,
                    tzinfo=UTC,
                ),
                data_updated_at="2026-08-14T23:00:00+0200",
                source="Eurostat",
                source_url="https://ec.europa.eu/eurostat/",
            ),
        ),
        "Question two": DeterministicAnswer(
            value=90.0,
            unit="NR",
            computation=ComputationProvenance(
                operation="none",
                input_values=(90.0,),
                input_time_periods=("2023",),
                input_dimensions=(),
                output_value=90.0,
            ),
            provenance=RetrievalProvenance(
                dataset_code="DEMO_PJAN",
                dataset_agency="ESTAT",
                dataset_version="1.0",
                filters=(("TIME_PERIOD", "2023"),),
                retrieved_at=datetime(
                    2026,
                    9,
                    13,
                    12,
                    0,
                    tzinfo=UTC,
                ),
                data_updated_at="2026-08-14T23:00:00+0200",
                source="Eurostat",
                source_url="https://ec.europa.eu/eurostat/",
            ),
        ),
    }

    def answer_question(question: str) -> DeterministicAnswer:
        return answers[question]

    run = run_benchmark(
        cases,
        answer_question,
    )

    assert len(run.results) == 2
    assert run.results[0].case == cases[0]
    assert run.results[1].case == cases[1]
    assert run.results[0].score.exact_match is True
    assert run.results[1].score.exact_match is True

    assert run.summary.total_cases == 2
    assert run.summary.dataset_accuracy == 1.0
    assert run.summary.filters_accuracy == 1.0
    assert run.summary.operation_accuracy == 1.0
    assert run.summary.exact_match_accuracy == 1.0


def test_run_benchmark_records_failed_case_without_stopping() -> None:
    cases = (
        BenchmarkCase(
            question="Working question",
            expected_dataset_code="DEMO_PJAN",
            expected_filters=(("TIME_PERIOD", "2024"),),
            expected_operation="none",
        ),
        BenchmarkCase(
            question="Failing question",
            expected_dataset_code="DEMO_PJAN",
            expected_filters=(("TIME_PERIOD", "2023"),),
            expected_operation="none",
        ),
    )

    working_answer = DeterministicAnswer(
        value=100.0,
        unit="NR",
        computation=ComputationProvenance(
            operation="none",
            input_values=(100.0,),
            input_time_periods=("2024",),
            input_dimensions=(),
            output_value=100.0,
        ),
        provenance=RetrievalProvenance(
            dataset_code="DEMO_PJAN",
            dataset_agency="ESTAT",
            dataset_version="1.0",
            filters=(("TIME_PERIOD", "2024"),),
            retrieved_at=datetime(
                2026,
                9,
                13,
                12,
                0,
                tzinfo=UTC,
            ),
            data_updated_at="2026-08-14T23:00:00+0200",
            source="Eurostat",
            source_url="https://ec.europa.eu/eurostat/",
        ),
    )

    def answer_question(question: str) -> DeterministicAnswer:
        if question == "Failing question":
            raise ValueError("Could not resolve dataset.")

        return working_answer

    run = run_benchmark(
        cases,
        answer_question,
    )

    assert len(run.results) == 1
    assert run.results[0].case == cases[0]

    assert len(run.failures) == 1
    assert run.failures[0] == BenchmarkFailure(
        case=cases[1],
        error_type="ValueError",
        error_message="Could not resolve dataset.",
    )

    assert run.summary.total_cases == 2
    assert run.summary.completed_cases == 1
    assert run.summary.failed_cases == 1


def test_run_benchmark_calculates_accuracy_over_completed_cases() -> None:
    cases = (
        BenchmarkCase(
            question="Working question",
            expected_dataset_code="DEMO_PJAN",
            expected_filters=(("TIME_PERIOD", "2024"),),
            expected_operation="none",
        ),
        BenchmarkCase(
            question="Failing question",
            expected_dataset_code="DEMO_PJAN",
            expected_filters=(("TIME_PERIOD", "2023"),),
            expected_operation="none",
        ),
    )

    working_answer = DeterministicAnswer(
        value=100.0,
        unit="NR",
        computation=ComputationProvenance(
            operation="none",
            input_values=(100.0,),
            input_time_periods=("2024",),
            input_dimensions=(),
            output_value=100.0,
        ),
        provenance=RetrievalProvenance(
            dataset_code="DEMO_PJAN",
            dataset_agency="ESTAT",
            dataset_version="1.0",
            filters=(("TIME_PERIOD", "2024"),),
            retrieved_at=datetime(
                2026,
                9,
                13,
                12,
                0,
                tzinfo=UTC,
            ),
            data_updated_at="2026-08-14T23:00:00+0200",
            source="Eurostat",
            source_url="https://ec.europa.eu/eurostat/",
        ),
    )

    def answer_question(question: str) -> DeterministicAnswer:
        if question == "Failing question":
            raise ValueError("Could not resolve dataset.")

        return working_answer

    run = run_benchmark(
        cases,
        answer_question,
    )

    assert run.summary.total_cases == 2
    assert run.summary.completed_cases == 1
    assert run.summary.failed_cases == 1

    assert run.summary.dataset_accuracy == 1.0
    assert run.summary.filters_accuracy == 1.0
    assert run.summary.operation_accuracy == 1.0
    assert run.summary.exact_match_accuracy == 1.0


def test_run_benchmark_handles_all_failed_cases() -> None:
    cases = (
        BenchmarkCase(
            question="First failing question",
            expected_dataset_code="DEMO_PJAN",
            expected_filters=(("TIME_PERIOD", "2024"),),
            expected_operation="none",
        ),
        BenchmarkCase(
            question="Second failing question",
            expected_dataset_code="DEMO_PJAN",
            expected_filters=(("TIME_PERIOD", "2023"),),
            expected_operation="none",
        ),
    )

    def answer_question(question: str) -> DeterministicAnswer:
        raise ValueError(f"Could not answer: {question}")

    run = run_benchmark(
        cases,
        answer_question,
    )

    assert run.results == ()
    assert len(run.failures) == 2

    assert run.summary.total_cases == 2
    assert run.summary.completed_cases == 0
    assert run.summary.failed_cases == 2

    assert run.summary.dataset_accuracy == 0.0
    assert run.summary.filters_accuracy == 0.0
    assert run.summary.operation_accuracy == 0.0
    assert run.summary.exact_match_accuracy == 0.0


def test_core_benchmark_contains_direct_population_lookup() -> None:
    assert CORE_BENCHMARK_CASES[0] == BenchmarkCase(
        question=("What was the population of 20-year-old women in Belgium in 2024?"),
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(
            ("TIME_PERIOD", "2024"),
            ("age", "Y20"),
            ("freq", "A"),
            ("geo", "BE"),
            ("sex", "F"),
            ("unit", "NR"),
        ),
        expected_operation="none",
    )


def test_core_benchmark_contains_population_paraphrase() -> None:
    assert CORE_BENCHMARK_CASES[1] == BenchmarkCase(
        question=("How many women aged 20 lived in Belgium on 1 January 2024?"),
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(
            ("TIME_PERIOD", "2024"),
            ("age", "Y20"),
            ("freq", "A"),
            ("geo", "BE"),
            ("sex", "F"),
            ("unit", "NR"),
        ),
        expected_operation="none",
    )


def test_core_benchmark_contains_different_population_codes() -> None:
    assert CORE_BENCHMARK_CASES[2] == BenchmarkCase(
        question=("What was the population of 20-year-old men in Germany in 2024?"),
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(
            ("TIME_PERIOD", "2024"),
            ("age", "Y20"),
            ("freq", "A"),
            ("geo", "DE"),
            ("sex", "M"),
            ("unit", "NR"),
        ),
        expected_operation="none",
    )


def test_core_benchmark_contains_sum_operation() -> None:
    assert CORE_BENCHMARK_CASES[3] == BenchmarkCase(
        question="What is the total population across 2023 and 2024?",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(),
        expected_operation="sum",
        score_filters=False,
    )


def test_run_benchmark_reports_completion_rate() -> None:
    case_one = BenchmarkCase(
        question="Question one",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(),
        expected_operation="none",
    )
    case_two = BenchmarkCase(
        question="Question two",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(),
        expected_operation="none",
    )

    answer = DeterministicAnswer(
        value=1.0,
        unit="NR",
        computation=ComputationProvenance(
            operation="none",
            input_values=(1.0,),
            input_time_periods=("2024",),
            input_dimensions=(),
            output_value=1.0,
        ),
        provenance=RetrievalProvenance(
            dataset_code="DEMO_PJAN",
            dataset_agency="ESTAT",
            dataset_version="72.0",
            filters=(),
            retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
            data_updated_at="2025-12-31",
            source="Eurostat",
            source_url="https://example.test",
        ),
    )

    def answer_question(question: str) -> DeterministicAnswer:
        if question == "Question two":
            raise RuntimeError("failed")

        return answer

    run = run_benchmark(
        (case_one, case_two),
        answer_question,
    )

    assert run.summary.completion_rate == 0.5


def test_benchmark_case_can_skip_filter_scoring() -> None:
    case = BenchmarkCase(
        question="What is the total population across 2023 and 2024?",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(),
        expected_operation="sum",
        score_filters=False,
    )

    assert case.score_filters is False


def test_score_benchmark_case_can_skip_filter_matching() -> None:
    case = BenchmarkCase(
        question="What is the total population across 2023 and 2024?",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(),
        expected_operation="sum",
        score_filters=False,
    )

    score = score_benchmark_case(
        case,
        actual_dataset_code="DEMO_PJAN",
        actual_filters=(("TIME_PERIOD", "2024"),),
        actual_operation="sum",
    )

    assert score.filters_match is True
    assert score.exact_match is True


def test_sum_benchmark_skips_filter_scoring() -> None:
    assert CORE_BENCHMARK_CASES[3].score_filters is False


def test_score_reports_filters_as_not_scored() -> None:
    case = BenchmarkCase(
        question="What is the total population across 2023 and 2024?",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(),
        expected_operation="sum",
        score_filters=False,
    )

    score = score_benchmark_case(
        case,
        actual_dataset_code="DEMO_PJAN",
        actual_filters=(("TIME_PERIOD", "2024"),),
        actual_operation="sum",
    )

    assert score.filters_scored is False


def test_summarize_benchmark_scores_excludes_unscored_filters() -> None:
    scores = (
        BenchmarkScore(
            dataset_match=True,
            filters_match=True,
            operation_match=True,
            exact_match=True,
            filters_scored=False,
        ),
        BenchmarkScore(
            dataset_match=True,
            filters_match=False,
            operation_match=True,
            exact_match=False,
            filters_scored=True,
        ),
    )

    summary = summarize_benchmark_scores(scores)

    assert summary.filters_accuracy == 0.0
