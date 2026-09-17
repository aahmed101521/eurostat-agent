from datetime import UTC, datetime

import pytest

from eurostat_agent.catalogue import DatasetIndex, DatasetRecord
from eurostat_agent.controller import QuestionPlan, answer_planned_question
from eurostat_agent.data import Observation
from eurostat_agent.metadata import (
    Codelist,
    CodelistRef,
    DatasetMetadata,
    DataStructure,
)
from eurostat_agent.tracing import TraceRecorder


def test_answer_planned_question_records_successful_execution_trace() -> None:
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

        def get_dataset_metadata(self, dataset_code: str) -> DatasetMetadata:
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

    recorder = TraceRecorder("execution-001")
    answer = answer_planned_question(
        FakePlanner(),
        FakeSelector(),
        FakeClient(),
        "What was the population in 2024?",
        index=index,
        retrieved_at=datetime(2026, 9, 11, 12, 0, tzinfo=UTC),
        trace_recorder=recorder,
        planning_metadata=(
            ("backend", "ollama"),
            ("model", "qwen3.5:latest"),
            ("prompt_variant", "constrained"),
        ),
        selection_metadata=(
            ("backend", "ollama"),
            ("model", "qwen3.5:latest"),
        ),
    )

    assert answer.value == 123.0
    assert answer.provenance.dataset_code == "DEMO_PJAN"
    assert recorder.trace.execution_id == "execution-001"
    assert tuple(event.stage for event in recorder.trace.events) == (
        "planning",
        "catalogue_search",
        "dataset_selection",
        "filter_resolution",
        "retrieval",
        "computation",
        "controller_total",
    )
    assert all(event.success for event in recorder.trace.events)
    assert all(event.duration_seconds >= 0.0 for event in recorder.trace.events)
    assert dict(recorder.trace.events[0].metadata) == {
        "backend": "ollama",
        "model": "qwen3.5:latest",
        "prompt_variant": "constrained",
    }
    assert dict(recorder.trace.events[2].metadata) == {
        "candidate_count": "1",
        "backend": "ollama",
        "model": "qwen3.5:latest",
    }


def test_answer_planned_question_records_failure_without_changing_exception() -> None:
    class FailingPlanner:
        def plan(self, question: str) -> QuestionPlan:
            raise ValueError("planner failed")

    class UnusedSelector:
        def select_dataset(
            self,
            question: str,
            candidates: tuple[DatasetRecord, ...],
        ) -> str:
            raise AssertionError("selector should not be called")

    class UnusedClient:
        def get_structure(self, dataset_code: str) -> DataStructure:
            raise AssertionError("client should not be called")

        def get_codelist(self, ref: CodelistRef) -> Codelist:
            raise AssertionError("client should not be called")

        def get_dataset_metadata(self, dataset_code: str) -> DatasetMetadata:
            raise AssertionError("client should not be called")

        def fetch_series(
            self,
            dataset_code: str,
            filters: dict[str, str],
        ) -> tuple[Observation, ...]:
            raise AssertionError("client should not be called")

    recorder = TraceRecorder("execution-failure")

    with pytest.raises(ValueError, match="planner failed"):
        answer_planned_question(
            FailingPlanner(),
            UnusedSelector(),
            UnusedClient(),
            "question",
            index=DatasetIndex(records=()),
            retrieved_at=datetime(2026, 9, 11, 12, 0, tzinfo=UTC),
            trace_recorder=recorder,
        )

    assert tuple(event.stage for event in recorder.trace.events) == (
        "planning",
        "controller_total",
    )
    assert recorder.trace.events[0].success is False
    assert recorder.trace.events[0].error_type == "ValueError"
    assert recorder.trace.events[0].error_message == "planner failed"
    assert recorder.trace.events[1].success is False
