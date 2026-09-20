from dataclasses import replace
from datetime import UTC, datetime

from eurostat_agent.catalogue import DatasetIndex, DatasetRecord
from eurostat_agent.controller import QuestionPlan
from eurostat_agent.data import Observation
from eurostat_agent.experiments import ExperimentConfig
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
from eurostat_agent.robustness import (
    RobustnessCase,
    has_complete_provenance,
    robustness_comparison_to_dict,
    run_robustness_case,
    summarize_robustness_evidence,
)


class FixedPlanner:
    def __init__(self, plan: QuestionPlan) -> None:
        self._plan = plan

    def plan(self, question: str) -> QuestionPlan:
        return self._plan


class FailingPlanner:
    def plan(self, question: str) -> QuestionPlan:
        raise ValueError("Planner returned invalid JSON.")


class FixedSelector:
    def select_dataset(
        self,
        question: str,
        candidates: tuple[DatasetRecord, ...],
    ) -> str:
        return "DEMO_PJAN"


class FakeClient:
    def __init__(self) -> None:
        self.structure_calls = 0
        self.codelist_calls = 0
        self.metadata_calls = 0
        self.series_calls = 0

        self.refs = {
            "freq": CodelistRef("ESTAT", "FREQ", "1.0"),
            "unit": CodelistRef("ESTAT", "UNIT", "1.0"),
            "age": CodelistRef("ESTAT", "AGE", "1.0"),
            "sex": CodelistRef("ESTAT", "SEX", "1.0"),
            "geo": CodelistRef("ESTAT", "GEO", "1.0"),
        }
        self.codelists = {
            self.refs["freq"]: Codelist(
                id="FREQ",
                agency="ESTAT",
                version="1.0",
                codes=(Code("A", "Annual"),),
            ),
            self.refs["unit"]: Codelist(
                id="UNIT",
                agency="ESTAT",
                version="1.0",
                codes=(Code("NR", "Number"),),
            ),
            self.refs["age"]: Codelist(
                id="AGE",
                agency="ESTAT",
                version="1.0",
                codes=(Code("Y20", "20 years"),),
            ),
            self.refs["sex"]: Codelist(
                id="SEX",
                agency="ESTAT",
                version="1.0",
                codes=(Code("F", "Females"),),
            ),
            self.refs["geo"]: Codelist(
                id="GEO",
                agency="ESTAT",
                version="1.0",
                codes=(Code("BE", "Belgium"),),
            ),
        }

    def get_structure(self, dataset_code: str) -> DataStructure:
        self.structure_calls += 1
        return DataStructure(
            id="DEMO_PJAN",
            agency="ESTAT",
            version="72.0",
            dimensions=(
                Dimension("freq", 1, self.refs["freq"]),
                Dimension("unit", 2, self.refs["unit"]),
                Dimension("age", 3, self.refs["age"]),
                Dimension("sex", 4, self.refs["sex"]),
                Dimension("geo", 5, self.refs["geo"]),
                Dimension("TIME_PERIOD", 6, None),
            ),
            measure_id="OBS_VALUE",
        )

    def get_codelist(self, ref: CodelistRef) -> Codelist:
        self.codelist_calls += 1
        return self.codelists[ref]

    def get_dataset_metadata(self, dataset_code: str) -> DatasetMetadata:
        self.metadata_calls += 1
        return DatasetMetadata(
            id="DEMO_PJAN",
            agency="ESTAT",
            version="72.0",
            data_updated_at="2026-08-14T23:00:00+0200",
        )

    def fetch_series(
        self,
        dataset_code: str,
        filters: dict[str, str],
    ) -> tuple[Observation, ...]:
        self.series_calls += 1
        assert filters == {
            "freq": "A",
            "unit": "NR",
            "age": "Y20",
            "sex": "F",
            "geo": "BE",
            "TIME_PERIOD": "2024",
        }
        return (
            Observation(
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
            ),
        )


def _index() -> DatasetIndex:
    return DatasetIndex(
        records=(
            DatasetRecord(
                code="DEMO_PJAN",
                title="Population on 1 January by age and sex",
                product_type="dataset",
                description=None,
                last_update="14.08.2026",
                last_modified=None,
                data_start="1960",
                data_end="2025",
                value_count=None,
                paths=(("Population",),),
            ),
        )
    )


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        name="test-constrained",
        model="test-model",
        prompt_variant="constrained",
    )


def _supported_case() -> RobustnessCase:
    return RobustnessCase(
        case_id="A01",
        category="direct_supported",
        question="What was the population of 20-year-old women in Belgium in 2024?",
        expected_capability="supported",
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


def _plan() -> QuestionPlan:
    return QuestionPlan(
        dataset_query="population by age and sex",
        filters={
            "freq": "Annual",
            "unit": "Number",
            "age": "20 years",
            "sex": "Females",
            "geo": "Belgium",
            "TIME_PERIOD": "2024",
        },
        operation="none",
    )


def test_run_robustness_case_preserves_complete_evidence() -> None:
    client = FakeClient()
    evidence = run_robustness_case(
        _supported_case(),
        config=_config(),
        planner=FixedPlanner(_plan()),
        selector=FixedSelector(),
        client=client,
        index=_index(),
        retrieved_at=datetime(2026, 9, 17, 12, 0, tzinfo=UTC),
    )

    assert evidence.outcome == "completed_exact"
    assert evidence.question_plan == _plan()
    assert evidence.selected_dataset_code == "DEMO_PJAN"
    assert evidence.answer is not None
    assert evidence.benchmark_score is not None
    assert evidence.benchmark_score.exact_match is True
    assert evidence.provenance_violation is False
    assert evidence.failure_stage is None
    assert evidence.operational_report.model_call_count == 2
    assert evidence.operational_report.cache_hits == 4
    assert evidence.operational_report.cache_misses == 6
    assert client.structure_calls == 1
    assert client.codelist_calls == 5
    assert client.metadata_calls == 1
    assert client.series_calls == 1


def test_supported_planner_failure_is_model_completion_failure() -> None:
    evidence = run_robustness_case(
        _supported_case(),
        config=_config(),
        planner=FailingPlanner(),
        selector=FixedSelector(),
        client=FakeClient(),
        index=_index(),
        retrieved_at=datetime(2026, 9, 17, 12, 0, tzinfo=UTC),
    )

    assert evidence.answer is None
    assert evidence.outcome == "model_completion_failure"
    assert evidence.failure_stage == "planner_model_call"
    assert evidence.operational_report.model_call_count == 1


def test_known_unsupported_model_failure_remains_model_failure() -> None:
    case = RobustnessCase(
        case_id="E01",
        category="multi_value_unsupported",
        question="Population in 2023 and 2024",
        expected_capability="known_unsupported",
        expected_dataset_code="DEMO_PJAN",
        expected_operation="none",
    )
    evidence = run_robustness_case(
        case,
        config=_config(),
        planner=FailingPlanner(),
        selector=FixedSelector(),
        client=FakeClient(),
        index=_index(),
        retrieved_at=datetime(2026, 9, 17, 12, 0, tzinfo=UTC),
    )

    assert evidence.outcome == "model_completion_failure"
    assert evidence.answer is None


def test_known_unsupported_with_plan_is_architecture_limitation() -> None:
    case = RobustnessCase(
        case_id="E01",
        category="multi_value_unsupported",
        question="Population in 2023 and 2024",
        expected_capability="known_unsupported",
        expected_dataset_code="DEMO_PJAN",
        expected_operation="none",
    )
    plan = QuestionPlan(
        dataset_query="population by age and sex",
        filters={"TIME_PERIOD": "2023 and 2024"},
        operation="none",
    )
    evidence = run_robustness_case(
        case,
        config=_config(),
        planner=FixedPlanner(plan),
        selector=FixedSelector(),
        client=FakeClient(),
        index=_index(),
        retrieved_at=datetime(2026, 9, 17, 12, 0, tzinfo=UTC),
    )

    assert evidence.outcome == "expected_architecture_limitation"
    assert evidence.question_plan == plan
    assert evidence.answer is None


def test_safe_failure_case_counts_visible_failure_as_safe() -> None:
    case = RobustnessCase(
        case_id="F01",
        category="safe_failure",
        question="Population of Atlantis in 2024",
        expected_capability="safe_failure",
    )
    evidence = run_robustness_case(
        case,
        config=_config(),
        planner=FailingPlanner(),
        selector=FixedSelector(),
        client=FakeClient(),
        index=_index(),
        retrieved_at=datetime(2026, 9, 17, 12, 0, tzinfo=UTC),
    )

    assert evidence.outcome == "safe_failure"
    assert evidence.answer is None
    assert evidence.provenance_violation is False


def test_has_complete_provenance_rejects_incomplete_answer() -> None:
    answer = DeterministicAnswer(
        value=1.0,
        unit="NR",
        computation=ComputationProvenance(
            operation="none",
            input_values=(1.0,),
            input_time_periods=("2024",),
            input_dimensions=((("geo", "BE"),),),
            output_value=1.0,
        ),
        provenance=RetrievalProvenance(
            dataset_code="DEMO_PJAN",
            dataset_agency="ESTAT",
            dataset_version="72.0",
            filters=(("geo", "BE"),),
            retrieved_at=datetime(2026, 9, 17, tzinfo=UTC),
            data_updated_at="2026-08-14",
            source="",
            source_url="https://example.test",
        ),
    )

    assert has_complete_provenance(answer) is False


def test_summary_separates_supported_architecture_and_safe_failure() -> None:
    supported = run_robustness_case(
        _supported_case(),
        config=_config(),
        planner=FixedPlanner(_plan()),
        selector=FixedSelector(),
        client=FakeClient(),
        index=_index(),
        retrieved_at=datetime(2026, 9, 17, 12, 0, tzinfo=UTC),
    )
    unsupported = run_robustness_case(
        RobustnessCase(
            case_id="E01",
            category="multi_value_unsupported",
            question="Population in 2023 and 2024",
            expected_capability="known_unsupported",
        ),
        config=_config(),
        planner=FixedPlanner(
            QuestionPlan(
                dataset_query="population by age and sex",
                filters={"TIME_PERIOD": "2023 and 2024"},
                operation="none",
            )
        ),
        selector=FixedSelector(),
        client=FakeClient(),
        index=_index(),
        retrieved_at=datetime(2026, 9, 17, 12, 0, tzinfo=UTC),
    )
    safe_failure = run_robustness_case(
        RobustnessCase(
            case_id="F01",
            category="safe_failure",
            question="Population of Atlantis in 2024",
            expected_capability="safe_failure",
        ),
        config=_config(),
        planner=FailingPlanner(),
        selector=FixedSelector(),
        client=FakeClient(),
        index=_index(),
        retrieved_at=datetime(2026, 9, 17, 12, 0, tzinfo=UTC),
    )

    summary = summarize_robustness_evidence((supported, unsupported, safe_failure))

    assert summary.total_cases == 3
    assert summary.completed_cases == 1
    assert summary.supported_cases == 1
    assert summary.completed_supported_cases == 1
    assert summary.exact_supported_cases == 1
    assert summary.supported_case_completion_rate == 1.0
    assert summary.supported_case_exact_match == 1.0
    assert summary.known_unsupported_cases == 1
    assert summary.architecture_limitation_count == 1
    assert summary.safe_failure_cases == 1
    assert summary.safe_failures == 1
    assert summary.safe_failure_rate == 1.0
    assert summary.provenance_violation_count == 0


def test_provenance_violation_is_not_counted_as_completion() -> None:
    evidence = run_robustness_case(
        _supported_case(),
        config=_config(),
        planner=FixedPlanner(_plan()),
        selector=FixedSelector(),
        client=FakeClient(),
        index=_index(),
        retrieved_at=datetime(2026, 9, 17, 12, 0, tzinfo=UTC),
    )
    violated = replace(
        evidence,
        outcome="unexpected_failure",
        provenance_violation=True,
    )

    summary = summarize_robustness_evidence((violated,))

    assert summary.completed_cases == 0
    assert summary.completed_supported_cases == 0
    assert summary.exact_supported_cases == 0
    assert summary.provenance_violation_count == 1


def test_temporal_summary_does_not_normalize_full_dates() -> None:
    annual_case = RobustnessCase(
        case_id="D01",
        category="temporal_semantics",
        question="Population in 2024",
        expected_capability="supported",
        expected_dataset_code="DEMO_PJAN",
        expected_operation="none",
        score_filters=False,
    )
    date_case = RobustnessCase(
        case_id="D02",
        category="temporal_semantics",
        question="Population on 1 January 2024",
        expected_capability="known_unsupported",
    )
    annual = run_robustness_case(
        annual_case,
        config=_config(),
        planner=FixedPlanner(
            QuestionPlan(
                dataset_query="population by age and sex",
                filters={
                    "freq": "Annual",
                    "unit": "Number",
                    "age": "20 years",
                    "sex": "Females",
                    "geo": "Belgium",
                    "TIME_PERIOD": "2024",
                },
                operation="none",
            )
        ),
        selector=FixedSelector(),
        client=FakeClient(),
        index=_index(),
        retrieved_at=datetime(2026, 9, 17, 12, 0, tzinfo=UTC),
    )
    full_date = run_robustness_case(
        date_case,
        config=_config(),
        planner=FixedPlanner(
            QuestionPlan(
                dataset_query="population by age and sex",
                filters={"TIME_PERIOD": "2024-01-01"},
                operation="none",
            )
        ),
        selector=FixedSelector(),
        client=FakeClient(),
        index=_index(),
        retrieved_at=datetime(2026, 9, 17, 12, 0, tzinfo=UTC),
    )

    summary = summarize_robustness_evidence((annual, full_date))
    assert summary.temporal_behavior.total_cases == 2
    assert summary.temporal_behavior.annual_year_values == 1
    assert summary.temporal_behavior.full_date_values == 1


def test_comparison_serialization_retains_case_level_evidence() -> None:
    evidence = run_robustness_case(
        _supported_case(),
        config=_config(),
        planner=FixedPlanner(_plan()),
        selector=FixedSelector(),
        client=FakeClient(),
        index=_index(),
        retrieved_at=datetime(2026, 9, 17, 12, 0, tzinfo=UTC),
    )
    summary = summarize_robustness_evidence((evidence,))

    from eurostat_agent.robustness import RobustnessComparison, RobustnessRun

    payload = robustness_comparison_to_dict(
        RobustnessComparison(
            runs=(
                RobustnessRun(
                    config=_config(),
                    evidence=(evidence,),
                    summary=summary,
                ),
            ),
            failures=(),
        )
    )

    run_payload = payload["runs"][0]
    assert run_payload["cases"][0]["case"]["case_id"] == "A01"
    assert run_payload["cases"][0]["execution_trace"]["execution_id"]
    assert run_payload["summary"]["provenance_violation_count"] == 0
