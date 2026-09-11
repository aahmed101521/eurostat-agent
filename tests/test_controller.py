from datetime import UTC, datetime

import pytest

from eurostat_agent.controller import (
    StructuredQuestion,
    answer_question,
    execute_question,
    plan_question,
    resolve_question_filters,
    retrieve_question,
)
from eurostat_agent.data import Observation
from eurostat_agent.metadata import (
    Code,
    Codelist,
    CodelistRef,
    DatasetMetadata,
    DataStructure,
    Dimension,
)


def test_structured_question_preserves_request() -> None:
    question = StructuredQuestion(
        dataset_code="DEMO_PJAN",
        filters={
            "geo": "Belgium",
            "sex": "female",
            "age": "20",
            "unit": "number",
            "freq": "annual",
            "TIME_PERIOD": "2024",
        },
        operation="none",
    )

    assert question.dataset_code == "DEMO_PJAN"
    assert question.filters["geo"] == "Belgium"
    assert question.filters["TIME_PERIOD"] == "2024"
    assert question.operation == "none"


def test_resolve_question_filters_resolves_human_label() -> None:
    geo_ref = CodelistRef(
        agency="ESTAT",
        id="GEO",
        version="28.0",
    )

    structure = DataStructure(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="72.0",
        dimensions=(
            Dimension(
                id="geo",
                position=1,
                codelist=geo_ref,
            ),
        ),
        measure_id="OBS_VALUE",
    )

    codelist = Codelist(
        id="GEO",
        agency="ESTAT",
        version="28.0",
        codes=(
            Code(id="BE", label="Belgium"),
            Code(id="DE", label="Germany"),
        ),
    )

    class FakeClient:
        def get_structure(self, dataset_code: str) -> DataStructure:
            assert dataset_code == "DEMO_PJAN"
            return structure

        def get_codelist(self, ref: CodelistRef) -> Codelist:
            assert ref == geo_ref
            return codelist

    question = StructuredQuestion(
        dataset_code="DEMO_PJAN",
        filters={"geo": "Belgium"},
        operation="none",
    )

    result = resolve_question_filters(FakeClient(), question)

    assert result == {"geo": "BE"}


def test_resolve_question_filters_preserves_time_period() -> None:
    question = StructuredQuestion(
        dataset_code="DEMO_PJAN",
        filters={"TIME_PERIOD": "2024"},
        operation="none",
    )

    class FakeClient:
        def get_structure(self, dataset_code: str) -> DataStructure:
            raise AssertionError("TIME_PERIOD should not require structure lookup")

        def get_codelist(self, ref: CodelistRef) -> Codelist:
            raise AssertionError("TIME_PERIOD should not require codelist lookup")

    result = resolve_question_filters(FakeClient(), question)

    assert result == {"TIME_PERIOD": "2024"}


def test_resolve_question_filters_handles_coded_and_time_filters() -> None:
    geo_ref = CodelistRef(
        agency="ESTAT",
        id="GEO",
        version="28.0",
    )

    structure = DataStructure(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="72.0",
        dimensions=(
            Dimension(
                id="geo",
                position=1,
                codelist=geo_ref,
            ),
        ),
        measure_id="OBS_VALUE",
    )

    codelist = Codelist(
        id="GEO",
        agency="ESTAT",
        version="28.0",
        codes=(
            Code(id="BE", label="Belgium"),
            Code(id="DE", label="Germany"),
        ),
    )

    class FakeClient:
        def get_structure(self, dataset_code: str) -> DataStructure:
            assert dataset_code == "DEMO_PJAN"
            return structure

        def get_codelist(self, ref: CodelistRef) -> Codelist:
            assert ref == geo_ref
            return codelist

    question = StructuredQuestion(
        dataset_code="DEMO_PJAN",
        filters={
            "geo": "Belgium",
            "TIME_PERIOD": "2024",
        },
        operation="none",
    )

    result = resolve_question_filters(FakeClient(), question)

    assert result == {
        "geo": "BE",
        "TIME_PERIOD": "2024",
    }


def test_retrieve_question_resolves_filters_and_retrieves_with_provenance() -> None:
    geo_ref = CodelistRef(
        agency="ESTAT",
        id="GEO",
        version="28.0",
    )

    structure = DataStructure(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="72.0",
        dimensions=(
            Dimension(
                id="geo",
                position=1,
                codelist=geo_ref,
            ),
        ),
        measure_id="OBS_VALUE",
    )

    codelist = Codelist(
        id="GEO",
        agency="ESTAT",
        version="28.0",
        codes=(
            Code(id="BE", label="Belgium"),
            Code(id="DE", label="Germany"),
        ),
    )

    metadata = DatasetMetadata(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="1.0",
        data_updated_at="2026-08-14T23:00:00+0200",
    )

    observation = Observation(
        dataset_code="DEMO_PJAN",
        dimensions={"geo": "BE"},
        time_period="2024",
        value=123.0,
    )

    class FakeClient:
        def get_structure(self, dataset_code: str) -> DataStructure:
            assert dataset_code == "DEMO_PJAN"
            return structure

        def get_codelist(self, ref: CodelistRef) -> Codelist:
            assert ref == geo_ref
            return codelist

        def get_dataset_metadata(self, dataset_code: str) -> DatasetMetadata:
            assert dataset_code == "DEMO_PJAN"
            return metadata

        def fetch_series(
            self,
            dataset_code: str,
            filters: dict[str, str],
        ) -> tuple[Observation, ...]:
            assert dataset_code == "DEMO_PJAN"
            assert filters == {
                "geo": "BE",
                "TIME_PERIOD": "2024",
            }
            return (observation,)

    question = StructuredQuestion(
        dataset_code="DEMO_PJAN",
        filters={
            "geo": "Belgium",
            "TIME_PERIOD": "2024",
        },
        operation="none",
    )

    retrieved_at = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)

    result = retrieve_question(
        FakeClient(),
        question,
        retrieved_at=retrieved_at,
    )

    assert result.observations == (observation,)
    assert result.provenance.dataset_code == "DEMO_PJAN"
    assert result.provenance.filters == (
        ("TIME_PERIOD", "2024"),
        ("geo", "BE"),
    )
    assert result.provenance.retrieved_at == retrieved_at


def test_execute_question_builds_direct_deterministic_answer() -> None:
    geo_ref = CodelistRef(
        agency="ESTAT",
        id="GEO",
        version="28.0",
    )

    structure = DataStructure(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="72.0",
        dimensions=(
            Dimension(
                id="geo",
                position=1,
                codelist=geo_ref,
            ),
        ),
        measure_id="OBS_VALUE",
    )

    codelist = Codelist(
        id="GEO",
        agency="ESTAT",
        version="28.0",
        codes=(
            Code(id="BE", label="Belgium"),
            Code(id="DE", label="Germany"),
        ),
    )

    metadata = DatasetMetadata(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="1.0",
        data_updated_at="2026-08-14T23:00:00+0200",
    )

    observation = Observation(
        dataset_code="DEMO_PJAN",
        dimensions={
            "geo": "BE",
            "unit": "NR",
        },
        time_period="2024",
        value=123.0,
    )

    class FakeClient:
        def get_structure(self, dataset_code: str) -> DataStructure:
            assert dataset_code == "DEMO_PJAN"
            return structure

        def get_codelist(self, ref: CodelistRef) -> Codelist:
            assert ref == geo_ref
            return codelist

        def get_dataset_metadata(self, dataset_code: str) -> DatasetMetadata:
            assert dataset_code == "DEMO_PJAN"
            return metadata

        def fetch_series(
            self,
            dataset_code: str,
            filters: dict[str, str],
        ) -> tuple[Observation, ...]:
            assert dataset_code == "DEMO_PJAN"
            assert filters == {
                "geo": "BE",
                "TIME_PERIOD": "2024",
            }
            return (observation,)

    question = StructuredQuestion(
        dataset_code="DEMO_PJAN",
        filters={
            "geo": "Belgium",
            "TIME_PERIOD": "2024",
        },
        operation="none",
    )

    retrieved_at = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)

    answer = execute_question(
        FakeClient(),
        question,
        retrieved_at=retrieved_at,
    )

    assert answer.value == 123.0
    assert answer.unit == "NR"
    assert answer.computation.operation == "none"
    assert answer.provenance.dataset_code == "DEMO_PJAN"
    assert answer.provenance.filters == (
        ("TIME_PERIOD", "2024"),
        ("geo", "BE"),
    )
    assert answer.provenance.retrieved_at == retrieved_at


def test_execute_question_dispatches_sum_computation() -> None:
    metadata = DatasetMetadata(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="1.0",
        data_updated_at="2026-08-14T23:00:00+0200",
    )

    observations = (
        Observation(
            dataset_code="DEMO_PJAN",
            dimensions={"unit": "NR"},
            time_period="2023",
            value=100.0,
        ),
        Observation(
            dataset_code="DEMO_PJAN",
            dimensions={"unit": "NR"},
            time_period="2024",
            value=150.0,
        ),
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
            assert filters == {}
            return observations

    question = StructuredQuestion(
        dataset_code="DEMO_PJAN",
        filters={},
        operation="sum",
    )

    retrieved_at = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)

    answer = execute_question(
        FakeClient(),
        question,
        retrieved_at=retrieved_at,
    )

    assert answer.value == 250.0
    assert answer.unit == "NR"
    assert answer.computation.operation == "sum"
    assert answer.computation.input_values == (100.0, 150.0)


def test_execute_question_propagates_unsupported_operation_error() -> None:
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
            raise AssertionError("No coded filters should be resolved")

        def get_codelist(self, ref: CodelistRef) -> Codelist:
            raise AssertionError("No coded filters should be resolved")

        def get_dataset_metadata(self, dataset_code: str) -> DatasetMetadata:
            assert dataset_code == "DEMO_PJAN"
            return metadata

        def fetch_series(
            self,
            dataset_code: str,
            filters: dict[str, str],
        ) -> tuple[Observation, ...]:
            assert dataset_code == "DEMO_PJAN"
            assert filters == {}
            return (observation,)

    question = StructuredQuestion(
        dataset_code="DEMO_PJAN",
        filters={},
        operation="multiply",
    )

    retrieved_at = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)

    with pytest.raises(
        ValueError,
        match="Unsupported computation operation: 'multiply'",
    ):
        execute_question(
            FakeClient(),
            question,
            retrieved_at=retrieved_at,
        )


def test_execute_question_resolves_complete_structured_request() -> None:
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

    class FakeClient:
        def get_structure(self, dataset_code: str) -> DataStructure:
            assert dataset_code == "DEMO_PJAN"
            return structure

        def get_codelist(self, ref: CodelistRef) -> Codelist:
            return codelists[ref.id]

        def get_dataset_metadata(self, dataset_code: str) -> DatasetMetadata:
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

    question = StructuredQuestion(
        dataset_code="DEMO_PJAN",
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

    retrieved_at = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)

    answer = execute_question(
        FakeClient(),
        question,
        retrieved_at=retrieved_at,
    )

    assert answer.value == 123.0
    assert answer.unit == "NR"
    assert answer.computation.operation == "none"
    assert answer.provenance.filters == (
        ("TIME_PERIOD", "2024"),
        ("age", "Y20"),
        ("freq", "A"),
        ("geo", "BE"),
        ("sex", "F"),
        ("unit", "NR"),
    )


def test_plan_question_uses_planner_to_create_structured_question() -> None:
    expected = StructuredQuestion(
        dataset_code="DEMO_PJAN",
        filters={
            "geo": "Belgium",
            "TIME_PERIOD": "2024",
        },
        operation="none",
    )

    class FakePlanner:
        def plan(self, question: str) -> StructuredQuestion:
            assert question == "What was the population in Belgium in 2024?"
            return expected

    result = plan_question(
        FakePlanner(),
        "What was the population in Belgium in 2024?",
    )

    assert result == expected


def test_answer_question_plans_and_executes() -> None:
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

    class FakePlanner:
        def plan(self, question: str) -> StructuredQuestion:
            assert question == "What was the population in 2024?"
            return StructuredQuestion(
                dataset_code="DEMO_PJAN",
                filters={"TIME_PERIOD": "2024"},
                operation="none",
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

    retrieved_at = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)

    answer = answer_question(
        FakePlanner(),
        FakeClient(),
        "What was the population in 2024?",
        retrieved_at=retrieved_at,
    )

    assert answer.value == 123.0
    assert answer.unit == "NR"
    assert answer.computation.operation == "none"
    assert answer.provenance.dataset_code == "DEMO_PJAN"
