from datetime import UTC, datetime

import pytest

from eurostat_agent.catalogue import DatasetIndex, DatasetRecord
from eurostat_agent.controller import (
    QuestionPlan,
    create_question_plan,
    execute_question,
    materialize_question_plan,
    select_dataset_candidate,
)
from eurostat_agent.data import Observation
from eurostat_agent.metadata import (
    Codelist,
    CodelistRef,
    DatasetMetadata,
    DataStructure,
)
from eurostat_agent.planner import JsonDatasetSelector, JsonPlanner


def test_json_planner_builds_question_plan() -> None:
    class FakeModel:
        def complete(self, prompt: str) -> str:
            assert "population in Belgium in 2024" in prompt
            return """
            {
                "dataset_query": "Population on 1 January by age and sex",
                "filters": {
                    "geo": "Belgium",
                    "TIME_PERIOD": "2024"
                },
                "operation": "none"
            }
            """

    planner = JsonPlanner(FakeModel())

    result = planner.plan("What was the population in Belgium in 2024?")

    assert result == QuestionPlan(
        dataset_query="Population on 1 January by age and sex",
        filters={
            "geo": "Belgium",
            "TIME_PERIOD": "2024",
        },
        operation="none",
    )


def test_json_planner_rejects_invalid_json() -> None:
    class FakeModel:
        def complete(self, prompt: str) -> str:
            return "this is not json"

    planner = JsonPlanner(FakeModel())

    with pytest.raises(
        ValueError,
        match="Planner returned invalid JSON.",
    ):
        planner.plan("What was the population in Belgium in 2024?")


def test_json_planner_rejects_missing_required_fields() -> None:
    class FakeModel:
        def complete(self, prompt: str) -> str:
            return """
            {
                "filters": {
                    "geo": "Belgium"
                },
                "operation": "none"
            }
            """

    planner = JsonPlanner(FakeModel())

    with pytest.raises(
        ValueError,
        match="Planner response is missing required fields.",
    ):
        planner.plan("What was the population in Belgium?")


def test_json_planner_rejects_wrong_field_types() -> None:
    class FakeModel:
        def complete(self, prompt: str) -> str:
            return """
            {
                "dataset_query": "Population on 1 January by age and sex",
                "filters": "Belgium",
                "operation": "none"
            }
            """

    planner = JsonPlanner(FakeModel())

    with pytest.raises(
        ValueError,
        match="Planner response has invalid field types.",
    ):
        planner.plan("What was the population in Belgium?")


def test_json_planner_rejects_unexpected_fields() -> None:
    class FakeModel:
        def complete(self, prompt: str) -> str:
            return """
            {
                "dataset_query": "Population on 1 January by age and sex",
                "filters": {
                    "geo": "Belgium"
                },
                "operation": "none",
                "explanation": "I chose a population search query."
            }
            """

    planner = JsonPlanner(FakeModel())

    with pytest.raises(
        ValueError,
        match="Planner response contains unexpected fields.",
    ):
        planner.plan("What was the population in Belgium?")


def test_json_planner_rejects_non_object_response() -> None:
    class FakeModel:
        def complete(self, prompt: str) -> str:
            return """
            [
                "Population on 1 January by age and sex",
                {"geo": "Belgium"},
                "none"
            ]
            """

    planner = JsonPlanner(FakeModel())

    with pytest.raises(
        ValueError,
        match="Planner response must be a JSON object.",
    ):
        planner.plan("What was the population in Belgium?")


def test_json_planner_drives_safe_deterministic_answer_path() -> None:
    class FakeModel:
        def complete(self, prompt: str) -> str:
            assert "population in 2024" in prompt
            return """
            {
                "dataset_query": "Population on 1 January by age and sex",
                "filters": {
                    "TIME_PERIOD": "2024"
                },
                "operation": "none"
            }
            """

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
        def get_structure(
            self,
            dataset_code: str,
        ) -> DataStructure:
            raise AssertionError("TIME_PERIOD should not require structure lookup")

        def get_codelist(
            self,
            ref: CodelistRef,
        ) -> Codelist:
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
            assert filters == {
                "TIME_PERIOD": "2024",
            }
            return (observation,)

    question = "What was the population in 2024?"

    planner = JsonPlanner(FakeModel())

    plan = planner.plan(question)

    structured_question = materialize_question_plan(
        FakeSelector(),
        plan,
        question=question,
        index=index,
    )

    retrieved_at = datetime(
        2026,
        9,
        11,
        12,
        0,
        tzinfo=UTC,
    )

    answer = execute_question(
        FakeClient(),
        structured_question,
        retrieved_at=retrieved_at,
    )

    assert structured_question.dataset_code == "DEMO_PJAN"
    assert answer.value == 123.0
    assert answer.unit == "NR"
    assert answer.computation.operation == "none"
    assert answer.provenance.dataset_code == "DEMO_PJAN"


def test_create_question_plan_uses_question_planner() -> None:
    expected = QuestionPlan(
        dataset_query="Population on 1 January by age and sex",
        filters={
            "geo": "Belgium",
            "TIME_PERIOD": "2024",
        },
        operation="none",
    )

    class FakePlanner:
        def plan(self, question: str) -> QuestionPlan:
            assert question == "What was the population in Belgium in 2024?"
            return expected

    result = create_question_plan(
        FakePlanner(),
        "What was the population in Belgium in 2024?",
    )

    assert result == expected


def test_json_dataset_selector_selects_from_candidates() -> None:
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

    migration = DatasetRecord(
        code="MIGR_IMM1CTZ",
        title="Immigration by age and citizenship",
        product_type="dataset",
        description=None,
        last_update=None,
        last_modified=None,
        data_start=None,
        data_end=None,
        value_count=None,
        paths=(),
    )

    class FakeModel:
        def complete(self, prompt: str) -> str:
            assert "DEMO_PJAN" in prompt
            assert "Population on 1 January by age and sex" in prompt
            assert "MIGR_IMM1CTZ" in prompt
            assert "Immigration by age and citizenship" in prompt

            return """
            {
                "dataset_code": "DEMO_PJAN"
            }
            """

    selector = JsonDatasetSelector(FakeModel())

    result = selector.select_dataset(
        "What was the population in Belgium in 2024?",
        (
            population,
            migration,
        ),
    )

    assert result == "DEMO_PJAN"


def test_json_dataset_selector_rejects_invalid_json() -> None:
    class FakeModel:
        def complete(self, prompt: str) -> str:
            return "not json"

    selector = JsonDatasetSelector(FakeModel())

    with pytest.raises(
        ValueError,
        match="Dataset selector returned invalid JSON.",
    ):
        selector.select_dataset(
            "What was the population in Belgium in 2024?",
            (),
        )


def test_json_dataset_selector_cannot_escape_discovered_candidates() -> None:
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

    class FakeModel:
        def complete(self, prompt: str) -> str:
            assert "DEMO_PJAN" in prompt
            return """
            {
                "dataset_code": "MADE_UP_DATASET"
            }
            """

    selector = JsonDatasetSelector(FakeModel())

    with pytest.raises(
        ValueError,
        match=(
            "Selected dataset 'MADE_UP_DATASET' is not among the discovered candidates."
        ),
    ):
        select_dataset_candidate(
            selector,
            "What was the population in Belgium in 2024?",
            (population,),
        )


def test_json_dataset_selector_rejects_missing_required_fields() -> None:
    class FakeModel:
        def complete(self, prompt: str) -> str:
            return """
            {
                "reason": "This looks like a population dataset."
            }
            """

    selector = JsonDatasetSelector(FakeModel())

    with pytest.raises(
        ValueError,
        match="Dataset selector response is missing required fields.",
    ):
        selector.select_dataset(
            "What was the population in Belgium in 2024?",
            (),
        )


def test_json_dataset_selector_rejects_unexpected_fields() -> None:
    class FakeModel:
        def complete(self, prompt: str) -> str:
            return """
            {
                "dataset_code": "DEMO_PJAN",
                "reason": "It matches the population question."
            }
            """

    selector = JsonDatasetSelector(FakeModel())

    with pytest.raises(
        ValueError,
        match="Dataset selector response contains unexpected fields.",
    ):
        selector.select_dataset(
            "What was the population in Belgium in 2024?",
            (),
        )


def test_json_dataset_selector_rejects_wrong_field_type() -> None:
    class FakeModel:
        def complete(self, prompt: str) -> str:
            return """
            {
                "dataset_code": 123
            }
            """

    selector = JsonDatasetSelector(FakeModel())

    with pytest.raises(
        ValueError,
        match="Dataset selector response has invalid field types.",
    ):
        selector.select_dataset(
            "What was the population in Belgium in 2024?",
            (),
        )


def test_json_dataset_selector_rejects_non_object_response() -> None:
    class FakeModel:
        def complete(self, prompt: str) -> str:
            return """
            [
                "DEMO_PJAN"
            ]
            """

    selector = JsonDatasetSelector(FakeModel())

    with pytest.raises(
        ValueError,
        match="Dataset selector response must be a JSON object.",
    ):
        selector.select_dataset(
            "What was the population in Belgium in 2024?",
            (),
        )
