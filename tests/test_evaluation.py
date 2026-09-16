import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

import eurostat_agent.evaluation as evaluation
from eurostat_agent.benchmark import CORE_BENCHMARK_CASES
from eurostat_agent.catalogue import DatasetIndex, DatasetRecord
from eurostat_agent.controller import QuestionPlan
from eurostat_agent.data import Observation
from eurostat_agent.evaluation import (
    BenchmarkCase,
    BenchmarkFailure,
    BenchmarkResult,
    BenchmarkRun,
    BenchmarkScore,
    BenchmarkSummary,
    benchmark_case_to_dict,
    benchmark_failure_to_dict,
    benchmark_result_to_dict,
    benchmark_run_to_dict,
    benchmark_summary_to_dict,
    build_benchmark_result,
    run_benchmark,
    score_benchmark_case,
    score_deterministic_answer,
    summarize_benchmark_scores,
    write_benchmark_report,
)
from eurostat_agent.metadata import (
    Code,
    Codelist,
    CodelistRef,
    DatasetMetadata,
    DataStructure,
    Dimension,
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


def test_run_controller_benchmark_routes_question_through_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = BenchmarkCase(
        question="What was the population of women in Belgium in 2024?",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(),
        expected_operation="none",
    )

    answer = DeterministicAnswer(
        value=123.0,
        unit="NR",
        computation=ComputationProvenance(
            operation="none",
            input_values=(123.0,),
            input_time_periods=("2024",),
            input_dimensions=(),
            output_value=123.0,
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

    planner = object()
    selector = object()
    client = object()
    index = object()
    retrieved_at = datetime(2026, 1, 1, tzinfo=UTC)

    calls: list[str] = []

    def fake_answer_planned_question(
        planner_arg: object,
        selector_arg: object,
        client_arg: object,
        question: str,
        *,
        index: object,
        retrieved_at: datetime,
        limit: int = 10,
    ) -> DeterministicAnswer:
        assert planner_arg is planner
        assert selector_arg is selector
        assert client_arg is client
        assert index is not None
        assert retrieved_at == datetime(2026, 1, 1, tzinfo=UTC)
        assert limit == 10

        calls.append(question)
        return answer

    monkeypatch.setattr(
        evaluation,
        "answer_planned_question",
        fake_answer_planned_question,
        raising=False,
    )

    run = evaluation.run_controller_benchmark(
        (case,),
        planner=planner,
        selector=selector,
        client=client,
        index=index,
        retrieved_at=retrieved_at,
    )

    assert calls == ["What was the population of women in Belgium in 2024?"]
    assert run.results[0].answer == answer
    assert run.summary.completed_cases == 1


def test_run_controller_benchmark_executes_real_controller_path() -> None:
    population = DatasetRecord(
        code="DEMO_PJAN",
        title="Population on 1 January by age and sex",
        product_type="dataset",
        description=None,
        last_update=None,
        last_modified=None,
        data_start=None,
        data_end=None,
        value_count=None,
        paths=(),
    )

    index = DatasetIndex(records=(population,))

    case = BenchmarkCase(
        question="What was the population in 2024?",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(("TIME_PERIOD", "2024"),),
        expected_operation="none",
    )

    class FakePlanner:
        def plan(self, question: str) -> QuestionPlan:
            assert question == "What was the population in 2024?"

            return QuestionPlan(
                dataset_query="Population on 1 January by age and sex",
                filters={"TIME_PERIOD": "2024"},
                operation="none",
            )

    class FakeSelector:
        def select_dataset(
            self,
            question: str,
            candidates: tuple[DatasetRecord, ...],
        ) -> str:
            assert question == "What was the population in 2024?"
            assert candidates == (population,)

            return "DEMO_PJAN"

    metadata = DatasetMetadata(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="1.0",
        data_updated_at="2026-08-14T23:00:00+0200",
    )

    observation = Observation(
        dataset_code="DEMO_PJAN",
        dimensions={"unit": "NR"},
        time_period="2024",
        value=123.0,
    )

    class FakeClient:
        def get_structure(self, dataset_code: str) -> DataStructure:
            raise AssertionError("TIME_PERIOD should not require structure lookup")

        def get_codelist(self, ref: CodelistRef) -> Codelist:
            raise AssertionError("TIME_PERIOD should not require codelist lookup")

        def get_dataset_metadata(
            self,
            dataset_code: str,
        ) -> DatasetMetadata:
            assert dataset_code == "DEMO_PJAN"

            return metadata

        def fetch_series(
            self,
            dataset_code: str,
            filters: dict[str, str],
        ) -> tuple[Observation, ...]:
            assert dataset_code == "DEMO_PJAN"
            assert filters == {"TIME_PERIOD": "2024"}

            return (observation,)

    run = evaluation.run_controller_benchmark(
        (case,),
        planner=FakePlanner(),
        selector=FakeSelector(),
        client=FakeClient(),
        index=index,
        retrieved_at=datetime(
            2026,
            9,
            11,
            12,
            0,
            tzinfo=UTC,
        ),
    )

    assert run.summary.total_cases == 1
    assert run.summary.completed_cases == 1
    assert run.summary.failed_cases == 0
    assert run.summary.completion_rate == 1.0
    assert run.summary.dataset_accuracy == 1.0
    assert run.summary.filters_accuracy == 1.0
    assert run.summary.operation_accuracy == 1.0
    assert run.summary.exact_match_accuracy == 1.0
    assert run.results[0].answer.value == 123.0
    assert run.results[0].answer.unit == "NR"


def test_core_benchmark_executes_human_label_resolution_path() -> None:
    population = DatasetRecord(
        code="DEMO_PJAN",
        title="Population on 1 January by age and sex",
        product_type="dataset",
        description=None,
        last_update=None,
        last_modified=None,
        data_start=None,
        data_end=None,
        value_count=None,
        paths=(),
    )

    index = DatasetIndex(records=(population,))

    refs = {
        "freq": CodelistRef("ESTAT", "FREQ", "1.0"),
        "unit": CodelistRef("ESTAT", "UNIT", "28.0"),
        "age": CodelistRef("ESTAT", "AGE", "2.0"),
        "sex": CodelistRef("ESTAT", "SEX", "2.0"),
        "geo": CodelistRef("ESTAT", "GEO", "28.0"),
    }

    structure = DataStructure(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="72.0",
        dimensions=(
            Dimension("freq", 1, refs["freq"]),
            Dimension("unit", 2, refs["unit"]),
            Dimension("age", 3, refs["age"]),
            Dimension("sex", 4, refs["sex"]),
            Dimension("geo", 5, refs["geo"]),
            Dimension("TIME_PERIOD", 6, None),
        ),
        measure_id="OBS_VALUE",
    )

    codelists = {
        "FREQ": Codelist(
            id="FREQ",
            agency="ESTAT",
            version="1.0",
            codes=(Code("A", "Annual"),),
        ),
        "UNIT": Codelist(
            id="UNIT",
            agency="ESTAT",
            version="28.0",
            codes=(Code("NR", "Number"),),
        ),
        "AGE": Codelist(
            id="AGE",
            agency="ESTAT",
            version="2.0",
            codes=(Code("Y20", "20 years"),),
        ),
        "SEX": Codelist(
            id="SEX",
            agency="ESTAT",
            version="2.0",
            codes=(Code("F", "Females"),),
        ),
        "GEO": Codelist(
            id="GEO",
            agency="ESTAT",
            version="28.0",
            codes=(Code("BE", "Belgium"),),
        ),
    }

    metadata = DatasetMetadata(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="1.0",
        data_updated_at="2026-08-14T23:00:00+0200",
    )

    observation = Observation(
        dataset_code="DEMO_PJAN",
        dimensions={
            "freq": "A",
            "unit": "NR",
            "age": "Y20",
            "sex": "F",
            "geo": "BE",
        },
        time_period="2024",
        value=123.0,
    )

    class FakePlanner:
        def plan(self, question: str) -> QuestionPlan:
            assert question == CORE_BENCHMARK_CASES[0].question

            return QuestionPlan(
                dataset_query="Population on 1 January by age and sex",
                filters={
                    "freq": "annual",
                    "unit": "number",
                    "age": "20",
                    "sex": "female",
                    "geo": "Belgium",
                    "TIME_PERIOD": "2024",
                },
                operation="none",
            )

    class FakeSelector:
        def select_dataset(
            self,
            question: str,
            candidates: tuple[DatasetRecord, ...],
        ) -> str:
            assert question == CORE_BENCHMARK_CASES[0].question
            assert candidates == (population,)

            return "DEMO_PJAN"

    class FakeClient:
        def get_structure(self, dataset_code: str) -> DataStructure:
            assert dataset_code == "DEMO_PJAN"
            return structure

        def get_codelist(self, ref: CodelistRef) -> Codelist:
            return codelists[ref.id]

        def get_dataset_metadata(
            self,
            dataset_code: str,
        ) -> DatasetMetadata:
            assert dataset_code == "DEMO_PJAN"
            return metadata

        def fetch_series(
            self,
            dataset_code: str,
            filters: dict[str, str],
        ) -> tuple[Observation, ...]:
            assert dataset_code == "DEMO_PJAN"
            assert filters == {
                "freq": "A",
                "unit": "NR",
                "age": "Y20",
                "sex": "F",
                "geo": "BE",
                "TIME_PERIOD": "2024",
            }

            return (observation,)

    run = evaluation.run_controller_benchmark(
        (CORE_BENCHMARK_CASES[0],),
        planner=FakePlanner(),
        selector=FakeSelector(),
        client=FakeClient(),
        index=index,
        retrieved_at=datetime(
            2026,
            9,
            11,
            12,
            0,
            tzinfo=UTC,
        ),
    )

    assert run.summary.completion_rate == 1.0
    assert run.summary.dataset_accuracy == 1.0
    assert run.summary.filters_accuracy == 1.0
    assert run.summary.operation_accuracy == 1.0
    assert run.summary.exact_match_accuracy == 1.0

    assert run.results[0].answer.provenance.filters == (
        ("TIME_PERIOD", "2024"),
        ("age", "Y20"),
        ("freq", "A"),
        ("geo", "BE"),
        ("sex", "F"),
        ("unit", "NR"),
    )


def test_controller_benchmark_runs_multiple_core_cases() -> None:
    population = DatasetRecord(
        code="DEMO_PJAN",
        title="Population on 1 January by age and sex",
        product_type="dataset",
        description=None,
        last_update=None,
        last_modified=None,
        data_start=None,
        data_end=None,
        value_count=None,
        paths=(),
    )

    index = DatasetIndex(records=(population,))

    refs = {
        "freq": CodelistRef("ESTAT", "FREQ", "1.0"),
        "unit": CodelistRef("ESTAT", "UNIT", "28.0"),
        "age": CodelistRef("ESTAT", "AGE", "2.0"),
        "sex": CodelistRef("ESTAT", "SEX", "2.0"),
        "geo": CodelistRef("ESTAT", "GEO", "28.0"),
    }

    structure = DataStructure(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="72.0",
        dimensions=(
            Dimension("freq", 1, refs["freq"]),
            Dimension("unit", 2, refs["unit"]),
            Dimension("age", 3, refs["age"]),
            Dimension("sex", 4, refs["sex"]),
            Dimension("geo", 5, refs["geo"]),
            Dimension("TIME_PERIOD", 6, None),
        ),
        measure_id="OBS_VALUE",
    )

    codelists = {
        "FREQ": Codelist(
            id="FREQ",
            agency="ESTAT",
            version="1.0",
            codes=(Code("A", "Annual"),),
        ),
        "UNIT": Codelist(
            id="UNIT",
            agency="ESTAT",
            version="28.0",
            codes=(Code("NR", "Number"),),
        ),
        "AGE": Codelist(
            id="AGE",
            agency="ESTAT",
            version="2.0",
            codes=(Code("Y20", "20 years"),),
        ),
        "SEX": Codelist(
            id="SEX",
            agency="ESTAT",
            version="2.0",
            codes=(Code("F", "Females"),),
        ),
        "GEO": Codelist(
            id="GEO",
            agency="ESTAT",
            version="28.0",
            codes=(Code("BE", "Belgium"),),
        ),
    }

    metadata = DatasetMetadata(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="1.0",
        data_updated_at="2026-08-14T23:00:00+0200",
    )

    observation = Observation(
        dataset_code="DEMO_PJAN",
        dimensions={
            "freq": "A",
            "unit": "NR",
            "age": "Y20",
            "sex": "F",
            "geo": "BE",
        },
        time_period="2024",
        value=123.0,
    )

    class FakePlanner:
        def plan(self, question: str) -> QuestionPlan:
            assert question in {
                CORE_BENCHMARK_CASES[0].question,
                CORE_BENCHMARK_CASES[1].question,
            }

            return QuestionPlan(
                dataset_query="Population on 1 January by age and sex",
                filters={
                    "freq": "annual",
                    "unit": "number",
                    "age": "20",
                    "sex": "female",
                    "geo": "Belgium",
                    "TIME_PERIOD": "2024",
                },
                operation="none",
            )

    class FakeSelector:
        def select_dataset(
            self,
            question: str,
            candidates: tuple[DatasetRecord, ...],
        ) -> str:
            assert question in {
                CORE_BENCHMARK_CASES[0].question,
                CORE_BENCHMARK_CASES[1].question,
            }
            assert candidates == (population,)

            return "DEMO_PJAN"

    class FakeClient:
        def get_structure(self, dataset_code: str) -> DataStructure:
            assert dataset_code == "DEMO_PJAN"
            return structure

        def get_codelist(self, ref: CodelistRef) -> Codelist:
            return codelists[ref.id]

        def get_dataset_metadata(
            self,
            dataset_code: str,
        ) -> DatasetMetadata:
            assert dataset_code == "DEMO_PJAN"
            return metadata

        def fetch_series(
            self,
            dataset_code: str,
            filters: dict[str, str],
        ) -> tuple[Observation, ...]:
            assert dataset_code == "DEMO_PJAN"
            assert filters == {
                "freq": "A",
                "unit": "NR",
                "age": "Y20",
                "sex": "F",
                "geo": "BE",
                "TIME_PERIOD": "2024",
            }

            return (observation,)

    run = evaluation.run_controller_benchmark(
        CORE_BENCHMARK_CASES[:2],
        planner=FakePlanner(),
        selector=FakeSelector(),
        client=FakeClient(),
        index=index,
        retrieved_at=datetime(
            2026,
            9,
            11,
            12,
            0,
            tzinfo=UTC,
        ),
    )

    assert run.summary.total_cases == 2
    assert run.summary.completed_cases == 2
    assert run.summary.failed_cases == 0
    assert run.summary.completion_rate == 1.0
    assert run.summary.dataset_accuracy == 1.0
    assert run.summary.filters_accuracy == 1.0
    assert run.summary.operation_accuracy == 1.0
    assert run.summary.exact_match_accuracy == 1.0
    assert len(run.results) == 2


def test_core_benchmark_executes_different_population_codes() -> None:
    population = DatasetRecord(
        code="DEMO_PJAN",
        title="Population on 1 January by age and sex",
        product_type="dataset",
        description=None,
        last_update=None,
        last_modified=None,
        data_start=None,
        data_end=None,
        value_count=None,
        paths=(),
    )

    index = DatasetIndex(records=(population,))

    refs = {
        "freq": CodelistRef("ESTAT", "FREQ", "1.0"),
        "unit": CodelistRef("ESTAT", "UNIT", "28.0"),
        "age": CodelistRef("ESTAT", "AGE", "2.0"),
        "sex": CodelistRef("ESTAT", "SEX", "2.0"),
        "geo": CodelistRef("ESTAT", "GEO", "28.0"),
    }

    structure = DataStructure(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="72.0",
        dimensions=(
            Dimension("freq", 1, refs["freq"]),
            Dimension("unit", 2, refs["unit"]),
            Dimension("age", 3, refs["age"]),
            Dimension("sex", 4, refs["sex"]),
            Dimension("geo", 5, refs["geo"]),
            Dimension("TIME_PERIOD", 6, None),
        ),
        measure_id="OBS_VALUE",
    )

    codelists = {
        "FREQ": Codelist(
            id="FREQ",
            agency="ESTAT",
            version="1.0",
            codes=(Code("A", "Annual"),),
        ),
        "UNIT": Codelist(
            id="UNIT",
            agency="ESTAT",
            version="28.0",
            codes=(Code("NR", "Number"),),
        ),
        "AGE": Codelist(
            id="AGE",
            agency="ESTAT",
            version="2.0",
            codes=(Code("Y20", "20 years"),),
        ),
        "SEX": Codelist(
            id="SEX",
            agency="ESTAT",
            version="2.0",
            codes=(
                Code("F", "Females"),
                Code("M", "Males"),
            ),
        ),
        "GEO": Codelist(
            id="GEO",
            agency="ESTAT",
            version="28.0",
            codes=(
                Code("BE", "Belgium"),
                Code("DE", "Germany"),
            ),
        ),
    }

    metadata = DatasetMetadata(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="1.0",
        data_updated_at="2026-08-14T23:00:00+0200",
    )

    observation = Observation(
        dataset_code="DEMO_PJAN",
        dimensions={
            "freq": "A",
            "unit": "NR",
            "age": "Y20",
            "sex": "M",
            "geo": "DE",
        },
        time_period="2024",
        value=456.0,
    )

    class FakePlanner:
        def plan(self, question: str) -> QuestionPlan:
            assert question == CORE_BENCHMARK_CASES[2].question

            return QuestionPlan(
                dataset_query="Population on 1 January by age and sex",
                filters={
                    "freq": "annual",
                    "unit": "number",
                    "age": "20",
                    "sex": "male",
                    "geo": "Germany",
                    "TIME_PERIOD": "2024",
                },
                operation="none",
            )

    class FakeSelector:
        def select_dataset(
            self,
            question: str,
            candidates: tuple[DatasetRecord, ...],
        ) -> str:
            assert question == CORE_BENCHMARK_CASES[2].question
            assert candidates == (population,)

            return "DEMO_PJAN"

    class FakeClient:
        def get_structure(self, dataset_code: str) -> DataStructure:
            assert dataset_code == "DEMO_PJAN"
            return structure

        def get_codelist(self, ref: CodelistRef) -> Codelist:
            return codelists[ref.id]

        def get_dataset_metadata(
            self,
            dataset_code: str,
        ) -> DatasetMetadata:
            assert dataset_code == "DEMO_PJAN"
            return metadata

        def fetch_series(
            self,
            dataset_code: str,
            filters: dict[str, str],
        ) -> tuple[Observation, ...]:
            assert dataset_code == "DEMO_PJAN"
            assert filters == {
                "freq": "A",
                "unit": "NR",
                "age": "Y20",
                "sex": "M",
                "geo": "DE",
                "TIME_PERIOD": "2024",
            }

            return (observation,)

    run = evaluation.run_controller_benchmark(
        (CORE_BENCHMARK_CASES[2],),
        planner=FakePlanner(),
        selector=FakeSelector(),
        client=FakeClient(),
        index=index,
        retrieved_at=datetime(
            2026,
            9,
            11,
            12,
            0,
            tzinfo=UTC,
        ),
    )

    assert run.summary.completed_cases == 1
    assert run.summary.failed_cases == 0
    assert run.summary.dataset_accuracy == 1.0
    assert run.summary.filters_accuracy == 1.0
    assert run.summary.operation_accuracy == 1.0
    assert run.summary.exact_match_accuracy == 1.0

    assert run.results[0].answer.value == 456.0
    assert run.results[0].answer.provenance.filters == (
        ("TIME_PERIOD", "2024"),
        ("age", "Y20"),
        ("freq", "A"),
        ("geo", "DE"),
        ("sex", "M"),
        ("unit", "NR"),
    )


def test_core_benchmark_reports_unsupported_multi_period_case() -> None:
    supported_questions = {
        CORE_BENCHMARK_CASES[0].question,
        CORE_BENCHMARK_CASES[1].question,
        CORE_BENCHMARK_CASES[2].question,
    }

    def answer_question(question: str) -> DeterministicAnswer:
        if question == CORE_BENCHMARK_CASES[3].question:
            raise ValueError(
                "Multi-period filters are not supported by the current question plan."
            )

        assert question in supported_questions

        return DeterministicAnswer(
            value=123.0,
            unit="NR",
            computation=ComputationProvenance(
                operation="none",
                input_values=(123.0,),
                input_time_periods=("2024",),
                input_dimensions=(),
                output_value=123.0,
            ),
            provenance=RetrievalProvenance(
                dataset_code="DEMO_PJAN",
                dataset_agency="ESTAT",
                dataset_version="72.0",
                filters=CORE_BENCHMARK_CASES[0].expected_filters,
                retrieved_at=datetime(2026, 9, 11, 12, 0, tzinfo=UTC),
                data_updated_at="2026-08-14T23:00:00+0200",
                source="Eurostat",
                source_url="https://example.test",
            ),
        )

    run = run_benchmark(
        CORE_BENCHMARK_CASES,
        answer_question,
    )

    assert run.summary.total_cases == 4
    assert run.summary.completed_cases == 3
    assert run.summary.failed_cases == 1
    assert run.summary.completion_rate == 0.75

    assert len(run.failures) == 1
    assert run.failures[0].case == CORE_BENCHMARK_CASES[3]
    assert run.failures[0].error_type == "ValueError"
    assert (
        run.failures[0].error_message
        == "Multi-period filters are not supported by the current question plan."
    )


def test_benchmark_summary_to_dict_returns_stable_metrics() -> None:
    summary = BenchmarkSummary(
        total_cases=4,
        completed_cases=3,
        failed_cases=1,
        completion_rate=0.75,
        dataset_accuracy=1.0,
        filters_accuracy=1.0,
        operation_accuracy=1.0,
        exact_match_accuracy=1.0,
    )

    assert benchmark_summary_to_dict(summary) == {
        "total_cases": 4,
        "completed_cases": 3,
        "failed_cases": 1,
        "completion_rate": 0.75,
        "dataset_accuracy": 1.0,
        "filters_accuracy": 1.0,
        "operation_accuracy": 1.0,
        "exact_match_accuracy": 1.0,
    }


def test_benchmark_run_to_dict_returns_stable_top_level_shape() -> None:
    summary = BenchmarkSummary(
        total_cases=0,
        completed_cases=0,
        failed_cases=0,
        completion_rate=0.0,
        dataset_accuracy=0.0,
        filters_accuracy=0.0,
        operation_accuracy=0.0,
        exact_match_accuracy=0.0,
    )

    run = BenchmarkRun(
        results=(),
        failures=(),
        summary=summary,
    )

    assert benchmark_run_to_dict(run) == {
        "schema_version": "1.0",
        "summary": {
            "total_cases": 0,
            "completed_cases": 0,
            "failed_cases": 0,
            "completion_rate": 0.0,
            "dataset_accuracy": 0.0,
            "filters_accuracy": 0.0,
            "operation_accuracy": 0.0,
            "exact_match_accuracy": 0.0,
        },
        "results": [],
        "failures": [],
    }


def test_benchmark_case_to_dict_returns_stable_semantics() -> None:
    case = BenchmarkCase(
        question="What was the population in 2024?",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(
            ("TIME_PERIOD", "2024"),
            ("geo", "BE"),
        ),
        expected_operation="none",
        score_filters=True,
    )

    assert benchmark_case_to_dict(case) == {
        "question": "What was the population in 2024?",
        "expected_dataset_code": "DEMO_PJAN",
        "expected_filters": [
            ["TIME_PERIOD", "2024"],
            ["geo", "BE"],
        ],
        "expected_operation": "none",
        "score_filters": True,
    }


def test_benchmark_failure_to_dict_returns_stable_error_details() -> None:
    case = BenchmarkCase(
        question="What is the total population across 2023 and 2024?",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(),
        expected_operation="sum",
        score_filters=False,
    )

    failure = BenchmarkFailure(
        case=case,
        error_type="ValueError",
        error_message="Multi-period filters are not supported.",
    )

    assert benchmark_failure_to_dict(failure) == {
        "case": {
            "question": "What is the total population across 2023 and 2024?",
            "expected_dataset_code": "DEMO_PJAN",
            "expected_filters": [],
            "expected_operation": "sum",
            "score_filters": False,
        },
        "error_type": "ValueError",
        "error_message": "Multi-period filters are not supported.",
    }


def test_benchmark_run_to_dict_includes_failures() -> None:
    case = BenchmarkCase(
        question="What is the total population across 2023 and 2024?",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(),
        expected_operation="sum",
        score_filters=False,
    )

    failure = BenchmarkFailure(
        case=case,
        error_type="ValueError",
        error_message="Multi-period filters are not supported.",
    )

    summary = BenchmarkSummary(
        total_cases=1,
        completed_cases=0,
        failed_cases=1,
        completion_rate=0.0,
        dataset_accuracy=0.0,
        filters_accuracy=0.0,
        operation_accuracy=0.0,
        exact_match_accuracy=0.0,
    )

    run = BenchmarkRun(
        results=(),
        failures=(failure,),
        summary=summary,
    )

    report = benchmark_run_to_dict(run)

    assert report["failures"] == [
        {
            "case": {
                "question": "What is the total population across 2023 and 2024?",
                "expected_dataset_code": "DEMO_PJAN",
                "expected_filters": [],
                "expected_operation": "sum",
                "score_filters": False,
            },
            "error_type": "ValueError",
            "error_message": "Multi-period filters are not supported.",
        }
    ]


def test_benchmark_result_to_dict_returns_stable_success_details() -> None:
    case = BenchmarkCase(
        question="What was the population in Belgium in 2024?",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(("TIME_PERIOD", "2024"),),
        expected_operation="none",
    )

    answer = DeterministicAnswer(
        value=123.0,
        unit="NR",
        computation=ComputationProvenance(
            operation="none",
            input_values=(123.0,),
            input_time_periods=("2024",),
            input_dimensions=(),
            output_value=123.0,
        ),
        provenance=RetrievalProvenance(
            dataset_code="DEMO_PJAN",
            dataset_agency="ESTAT",
            dataset_version="72.0",
            filters=(("TIME_PERIOD", "2024"),),
            retrieved_at=datetime(2026, 9, 11, 12, 0, tzinfo=UTC),
            data_updated_at="2026-08-14T23:00:00+0200",
            source="Eurostat",
            source_url="https://example.test",
        ),
    )

    result = build_benchmark_result(case, answer)

    assert benchmark_result_to_dict(result) == {
        "case": {
            "question": "What was the population in Belgium in 2024?",
            "expected_dataset_code": "DEMO_PJAN",
            "expected_filters": [["TIME_PERIOD", "2024"]],
            "expected_operation": "none",
            "score_filters": True,
        },
        "score": {
            "dataset_match": True,
            "filters_match": True,
            "operation_match": True,
            "exact_match": True,
            "filters_scored": True,
        },
        "answer": {
            "value": 123.0,
            "unit": "NR",
            "dataset_code": "DEMO_PJAN",
            "filters": [["TIME_PERIOD", "2024"]],
            "operation": "none",
        },
    }


def test_benchmark_run_to_dict_includes_results() -> None:
    case = BenchmarkCase(
        question="What was the population in Belgium in 2024?",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(("TIME_PERIOD", "2024"),),
        expected_operation="none",
    )

    answer = DeterministicAnswer(
        value=123.0,
        unit="NR",
        computation=ComputationProvenance(
            operation="none",
            input_values=(123.0,),
            input_time_periods=("2024",),
            input_dimensions=(),
            output_value=123.0,
        ),
        provenance=RetrievalProvenance(
            dataset_code="DEMO_PJAN",
            dataset_agency="ESTAT",
            dataset_version="72.0",
            filters=(("TIME_PERIOD", "2024"),),
            retrieved_at=datetime(
                2026,
                9,
                11,
                12,
                0,
                tzinfo=UTC,
            ),
            data_updated_at="2026-08-14T23:00:00+0200",
            source="Eurostat",
            source_url="https://example.test",
        ),
    )

    result = build_benchmark_result(
        case,
        answer,
    )

    summary = BenchmarkSummary(
        total_cases=1,
        completed_cases=1,
        failed_cases=0,
        completion_rate=1.0,
        dataset_accuracy=1.0,
        filters_accuracy=1.0,
        operation_accuracy=1.0,
        exact_match_accuracy=1.0,
    )

    run = BenchmarkRun(
        results=(result,),
        failures=(),
        summary=summary,
    )

    report = benchmark_run_to_dict(run)

    assert report["results"] == [
        {
            "case": {
                "question": "What was the population in Belgium in 2024?",
                "expected_dataset_code": "DEMO_PJAN",
                "expected_filters": [["TIME_PERIOD", "2024"]],
                "expected_operation": "none",
                "score_filters": True,
            },
            "score": {
                "dataset_match": True,
                "filters_match": True,
                "operation_match": True,
                "exact_match": True,
                "filters_scored": True,
            },
            "answer": {
                "value": 123.0,
                "unit": "NR",
                "dataset_code": "DEMO_PJAN",
                "filters": [["TIME_PERIOD", "2024"]],
                "operation": "none",
            },
        }
    ]


def test_write_benchmark_report_creates_json_file(
    tmp_path: Path,
) -> None:
    summary = BenchmarkSummary(
        total_cases=0,
        completed_cases=0,
        failed_cases=0,
        completion_rate=0.0,
        dataset_accuracy=0.0,
        filters_accuracy=0.0,
        operation_accuracy=0.0,
        exact_match_accuracy=0.0,
    )

    run = BenchmarkRun(
        results=(),
        failures=(),
        summary=summary,
    )

    report_path = tmp_path / "benchmark.json"

    write_benchmark_report(
        run,
        report_path,
    )

    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report == {
        "schema_version": "1.0",
        "summary": {
            "total_cases": 0,
            "completed_cases": 0,
            "failed_cases": 0,
            "completion_rate": 0.0,
            "dataset_accuracy": 0.0,
            "filters_accuracy": 0.0,
            "operation_accuracy": 0.0,
            "exact_match_accuracy": 0.0,
        },
        "results": [],
        "failures": [],
    }


def test_benchmark_run_to_dict_includes_schema_version() -> None:
    summary = BenchmarkSummary(
        total_cases=0,
        completed_cases=0,
        failed_cases=0,
        completion_rate=0.0,
        dataset_accuracy=0.0,
        filters_accuracy=0.0,
        operation_accuracy=0.0,
        exact_match_accuracy=0.0,
    )

    run = BenchmarkRun(
        results=(),
        failures=(),
        summary=summary,
    )

    report = benchmark_run_to_dict(run)

    assert report["schema_version"] == "1.0"
