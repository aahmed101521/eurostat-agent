from datetime import UTC, datetime

import pytest

from eurostat_agent.controller import StructuredQuestion, answer_question
from eurostat_agent.data import Observation
from eurostat_agent.metadata import (
    Codelist,
    CodelistRef,
    DatasetMetadata,
    DataStructure,
)
from eurostat_agent.planner import JsonPlanner


def test_json_planner_builds_structured_question() -> None:
    class FakeModel:
        def complete(self, prompt: str) -> str:
            assert "population in Belgium in 2024" in prompt
            return """
            {
                "dataset_code": "DEMO_PJAN",
                "filters": {
                    "geo": "Belgium",
                    "TIME_PERIOD": "2024"
                },
                "operation": "none"
            }
            """

    planner = JsonPlanner(FakeModel())

    result = planner.plan("What was the population in Belgium in 2024?")

    assert result == StructuredQuestion(
        dataset_code="DEMO_PJAN",
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
                "dataset_code": "DEMO_PJAN",
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
                "dataset_code": "DEMO_PJAN",
                "filters": {
                    "geo": "Belgium"
                },
                "operation": "none",
                "explanation": "I chose the population dataset."
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
                "DEMO_PJAN",
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


def test_json_planner_drives_deterministic_answer_path() -> None:
    class FakeModel:
        def complete(self, prompt: str) -> str:
            assert "population in 2024" in prompt
            return """
            {
                "dataset_code": "DEMO_PJAN",
                "filters": {
                    "TIME_PERIOD": "2024"
                },
                "operation": "none"
            }
            """

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

    planner = JsonPlanner(FakeModel())
    retrieved_at = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)

    answer = answer_question(
        planner,
        FakeClient(),
        "What was the population in 2024?",
        retrieved_at=retrieved_at,
    )

    assert answer.value == 123.0
    assert answer.unit == "NR"
    assert answer.computation.operation == "none"
    assert answer.provenance.dataset_code == "DEMO_PJAN"
