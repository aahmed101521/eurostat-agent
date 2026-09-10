from datetime import UTC, datetime

import pytest

from eurostat_agent.data import Observation
from eurostat_agent.metadata import DatasetMetadata
from eurostat_agent.provenance import (
    build_computed_answer,
    build_deterministic_answer,
    build_retrieval_provenance,
    build_retrieval_result,
    retrieve_with_provenance,
)


def test_build_retrieval_provenance_preserves_exact_query_context() -> None:
    metadata = DatasetMetadata(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="1.0",
        data_updated_at="2026-08-14T23:00:00+0200",
    )

    retrieved_at = datetime(
        2026,
        9,
        10,
        12,
        30,
        tzinfo=UTC,
    )

    provenance = build_retrieval_provenance(
        dataset_code="DEMO_PJAN",
        filters={
            "sex": "F",
            "geo": "BE",
            "age": "Y20",
            "freq": "A",
            "TIME_PERIOD": "2024",
        },
        metadata=metadata,
        retrieved_at=retrieved_at,
    )

    assert provenance.dataset_code == "DEMO_PJAN"
    assert provenance.dataset_agency == "ESTAT"
    assert provenance.dataset_version == "1.0"
    assert provenance.filters == (
        ("TIME_PERIOD", "2024"),
        ("age", "Y20"),
        ("freq", "A"),
        ("geo", "BE"),
        ("sex", "F"),
    )
    assert provenance.retrieved_at == retrieved_at
    assert provenance.data_updated_at == "2026-08-14T23:00:00+0200"
    assert provenance.source == "Eurostat SDMX 3.0"
    assert provenance.source_url == (
        "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/"
        "data/dataflow/ESTAT/DEMO_PJAN/1.0"
    )


def test_build_retrieval_provenance_rejects_mismatched_dataset_metadata() -> None:
    metadata = DatasetMetadata(
        id="OTHER_DATASET",
        agency="ESTAT",
        version="1.0",
        data_updated_at="2026-08-14T23:00:00+0200",
    )

    retrieved_at = datetime(
        2026,
        9,
        10,
        12,
        30,
        tzinfo=UTC,
    )

    with pytest.raises(ValueError, match="does not match"):
        build_retrieval_provenance(
            dataset_code="DEMO_PJAN",
            filters={
                "geo": "BE",
            },
            metadata=metadata,
            retrieved_at=retrieved_at,
        )


def test_build_retrieval_provenance_rejects_naive_retrieval_time() -> None:
    metadata = DatasetMetadata(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="1.0",
        data_updated_at="2026-08-14T23:00:00+0200",
    )

    retrieved_at = datetime(
        2026,
        9,
        10,
        12,
        30,
    )

    with pytest.raises(ValueError, match="timezone-aware"):
        build_retrieval_provenance(
            dataset_code="DEMO_PJAN",
            filters={
                "geo": "BE",
            },
            metadata=metadata,
            retrieved_at=retrieved_at,
        )


def test_build_retrieval_result_preserves_observation_and_provenance() -> None:
    metadata = DatasetMetadata(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="1.0",
        data_updated_at="2026-08-14T23:00:00+0200",
    )

    retrieved_at = datetime(
        2026,
        9,
        10,
        12,
        30,
        tzinfo=UTC,
    )

    provenance = build_retrieval_provenance(
        dataset_code="DEMO_PJAN",
        filters={
            "freq": "A",
            "age": "Y20",
            "sex": "F",
            "geo": "BE",
            "TIME_PERIOD": "2024",
        },
        metadata=metadata,
        retrieved_at=retrieved_at,
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
        value=65231.0,
    )

    result = build_retrieval_result(
        observations=(observation,),
        provenance=provenance,
    )

    assert result.observations == (observation,)
    assert result.provenance == provenance


def test_build_retrieval_result_rejects_observation_from_wrong_dataset() -> None:
    metadata = DatasetMetadata(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="1.0",
        data_updated_at="2026-08-14T23:00:00+0200",
    )

    provenance = build_retrieval_provenance(
        dataset_code="DEMO_PJAN",
        filters={
            "geo": "BE",
        },
        metadata=metadata,
        retrieved_at=datetime(
            2026,
            9,
            10,
            12,
            30,
            tzinfo=UTC,
        ),
    )

    observation = Observation(
        dataset_code="OTHER_DATASET",
        dimensions={
            "geo": "BE",
        },
        time_period="2024",
        value=1.0,
    )

    with pytest.raises(ValueError, match="does not match provenance dataset"):
        build_retrieval_result(
            observations=(observation,),
            provenance=provenance,
        )


def test_build_retrieval_result_rejects_observation_that_violates_filters() -> None:
    metadata = DatasetMetadata(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="1.0",
        data_updated_at="2026-08-14T23:00:00+0200",
    )

    provenance = build_retrieval_provenance(
        dataset_code="DEMO_PJAN",
        filters={
            "geo": "BE",
            "sex": "F",
            "TIME_PERIOD": "2024",
        },
        metadata=metadata,
        retrieved_at=datetime(
            2026,
            9,
            10,
            12,
            30,
            tzinfo=UTC,
        ),
    )

    observation = Observation(
        dataset_code="DEMO_PJAN",
        dimensions={
            "geo": "DE",
            "sex": "F",
            "unit": "NR",
        },
        time_period="2024",
        value=65231.0,
    )

    with pytest.raises(ValueError, match="does not satisfy provenance filter"):
        build_retrieval_result(
            observations=(observation,),
            provenance=provenance,
        )


def test_retrieve_with_provenance_builds_complete_retrieval_result() -> None:
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
        value=65231.0,
    )

    class FakeClient:
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
                "age": "Y20",
                "sex": "F",
                "geo": "BE",
                "TIME_PERIOD": "2024",
            }
            return (observation,)

    retrieved_at = datetime(
        2026,
        9,
        10,
        12,
        30,
        tzinfo=UTC,
    )

    result = retrieve_with_provenance(
        FakeClient(),
        dataset_code="DEMO_PJAN",
        filters={
            "freq": "A",
            "age": "Y20",
            "sex": "F",
            "geo": "BE",
            "TIME_PERIOD": "2024",
        },
        retrieved_at=retrieved_at,
    )

    assert result.observations == (observation,)
    assert result.provenance.dataset_code == "DEMO_PJAN"
    assert result.provenance.retrieved_at == retrieved_at
    assert result.provenance.data_updated_at == "2026-08-14T23:00:00+0200"


def test_build_deterministic_answer_from_single_observation() -> None:
    metadata = DatasetMetadata(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="1.0",
        data_updated_at="2026-08-14T23:00:00+0200",
    )

    provenance = build_retrieval_provenance(
        dataset_code="DEMO_PJAN",
        filters={
            "freq": "A",
            "age": "Y20",
            "sex": "F",
            "geo": "BE",
            "TIME_PERIOD": "2024",
        },
        metadata=metadata,
        retrieved_at=datetime(
            2026,
            9,
            10,
            12,
            30,
            tzinfo=UTC,
        ),
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
        value=65231.0,
    )

    retrieval = build_retrieval_result(
        observations=(observation,),
        provenance=provenance,
    )

    answer = build_deterministic_answer(retrieval)

    assert answer.value == 65231.0
    assert answer.unit == "NR"
    assert answer.computation.operation == "none"
    assert answer.computation.input_values == (65231.0,)
    assert answer.computation.output_value == 65231.0
    assert answer.provenance == provenance


def test_build_deterministic_answer_rejects_multiple_observations() -> None:
    metadata = DatasetMetadata(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="1.0",
        data_updated_at="2026-08-14T23:00:00+0200",
    )

    provenance = build_retrieval_provenance(
        dataset_code="DEMO_PJAN",
        filters={
            "geo": "BE",
        },
        metadata=metadata,
        retrieved_at=datetime(
            2026,
            9,
            10,
            12,
            30,
            tzinfo=UTC,
        ),
    )

    observations = (
        Observation(
            dataset_code="DEMO_PJAN",
            dimensions={
                "geo": "BE",
                "unit": "NR",
            },
            time_period="2023",
            value=100.0,
        ),
        Observation(
            dataset_code="DEMO_PJAN",
            dimensions={
                "geo": "BE",
                "unit": "NR",
            },
            time_period="2024",
            value=110.0,
        ),
    )

    retrieval = build_retrieval_result(
        observations=observations,
        provenance=provenance,
    )

    with pytest.raises(
        ValueError,
        match="requires exactly one observation",
    ):
        build_deterministic_answer(retrieval)


def test_build_deterministic_answer_rejects_missing_unit() -> None:
    metadata = DatasetMetadata(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="1.0",
        data_updated_at="2026-08-14T23:00:00+0200",
    )

    provenance = build_retrieval_provenance(
        dataset_code="DEMO_PJAN",
        filters={
            "geo": "BE",
            "TIME_PERIOD": "2024",
        },
        metadata=metadata,
        retrieved_at=datetime(
            2026,
            9,
            10,
            12,
            30,
            tzinfo=UTC,
        ),
    )

    observation = Observation(
        dataset_code="DEMO_PJAN",
        dimensions={
            "geo": "BE",
        },
        time_period="2024",
        value=65231.0,
    )

    retrieval = build_retrieval_result(
        observations=(observation,),
        provenance=provenance,
    )

    with pytest.raises(
        ValueError,
        match="does not contain a unit dimension",
    ):
        build_deterministic_answer(retrieval)


def test_build_computed_answer_sums_observations() -> None:
    metadata = DatasetMetadata(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="1.0",
        data_updated_at="2026-08-14T23:00:00+0200",
    )

    provenance = build_retrieval_provenance(
        dataset_code="DEMO_PJAN",
        filters={
            "freq": "A",
            "age": "Y20",
            "sex": "F",
            "TIME_PERIOD": "2024",
        },
        metadata=metadata,
        retrieved_at=datetime(
            2026,
            9,
            10,
            12,
            30,
            tzinfo=UTC,
        ),
    )

    observations = (
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
            value=100.0,
        ),
        Observation(
            dataset_code="DEMO_PJAN",
            dimensions={
                "freq": "A",
                "unit": "NR",
                "age": "Y20",
                "sex": "F",
                "geo": "DE",
            },
            time_period="2024",
            value=150.0,
        ),
    )

    retrieval = build_retrieval_result(
        observations=observations,
        provenance=provenance,
    )

    answer = build_computed_answer(
        retrieval,
        operation="sum",
    )

    assert answer.value == 250.0
    assert answer.unit == "NR"
    assert answer.computation.operation == "sum"
    assert answer.computation.input_values == (100.0, 150.0)
    assert answer.computation.output_value == 250.0
    assert answer.provenance == provenance


def test_build_computed_answer_rejects_mixed_units() -> None:
    metadata = DatasetMetadata(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="1.0",
        data_updated_at="2026-08-14T23:00:00+0200",
    )

    provenance = build_retrieval_provenance(
        dataset_code="DEMO_PJAN",
        filters={
            "freq": "A",
            "age": "Y20",
            "sex": "F",
            "TIME_PERIOD": "2024",
        },
        metadata=metadata,
        retrieved_at=datetime(
            2026,
            9,
            10,
            12,
            30,
            tzinfo=UTC,
        ),
    )

    observations = (
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
            value=100.0,
        ),
        Observation(
            dataset_code="DEMO_PJAN",
            dimensions={
                "freq": "A",
                "unit": "PC",
                "age": "Y20",
                "sex": "F",
                "geo": "DE",
            },
            time_period="2024",
            value=50.0,
        ),
    )

    retrieval = build_retrieval_result(
        observations=observations,
        provenance=provenance,
    )

    with pytest.raises(
        ValueError,
        match="same unit",
    ):
        build_computed_answer(
            retrieval,
            operation="sum",
        )


def test_build_computed_answer_calculates_difference() -> None:
    metadata = DatasetMetadata(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="1.0",
        data_updated_at="2026-08-14T23:00:00+0200",
    )

    provenance = build_retrieval_provenance(
        dataset_code="DEMO_PJAN",
        filters={
            "freq": "A",
            "age": "Y20",
            "sex": "F",
            "geo": "BE",
        },
        metadata=metadata,
        retrieved_at=datetime(
            2026,
            9,
            10,
            12,
            30,
            tzinfo=UTC,
        ),
    )

    observations = (
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
            value=150.0,
        ),
        Observation(
            dataset_code="DEMO_PJAN",
            dimensions={
                "freq": "A",
                "unit": "NR",
                "age": "Y20",
                "sex": "F",
                "geo": "BE",
            },
            time_period="2023",
            value=100.0,
        ),
    )

    retrieval = build_retrieval_result(
        observations=observations,
        provenance=provenance,
    )

    answer = build_computed_answer(
        retrieval,
        operation="difference",
    )

    assert answer.value == 50.0
    assert answer.unit == "NR"
    assert answer.computation.operation == "difference"
    assert answer.computation.input_values == (150.0, 100.0)
    assert answer.computation.input_time_periods == (
        "2024",
        "2023",
    )
    assert answer.computation.input_dimensions == (
        (
            ("age", "Y20"),
            ("freq", "A"),
            ("geo", "BE"),
            ("sex", "F"),
            ("unit", "NR"),
        ),
        (
            ("age", "Y20"),
            ("freq", "A"),
            ("geo", "BE"),
            ("sex", "F"),
            ("unit", "NR"),
        ),
    )
    assert answer.computation.output_value == 50.0
    assert answer.provenance == provenance


def test_build_computed_answer_difference_requires_two_observations() -> None:
    metadata = DatasetMetadata(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="1.0",
        data_updated_at="2026-08-14T23:00:00+0200",
    )

    provenance = build_retrieval_provenance(
        dataset_code="DEMO_PJAN",
        filters={
            "freq": "A",
            "age": "Y20",
            "sex": "F",
            "geo": "BE",
        },
        metadata=metadata,
        retrieved_at=datetime(
            2026,
            9,
            10,
            12,
            30,
            tzinfo=UTC,
        ),
    )

    observations = (
        Observation(
            dataset_code="DEMO_PJAN",
            dimensions={
                "freq": "A",
                "unit": "NR",
                "age": "Y20",
                "sex": "F",
                "geo": "BE",
            },
            time_period="2022",
            value=90.0,
        ),
        Observation(
            dataset_code="DEMO_PJAN",
            dimensions={
                "freq": "A",
                "unit": "NR",
                "age": "Y20",
                "sex": "F",
                "geo": "BE",
            },
            time_period="2023",
            value=100.0,
        ),
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
            value=110.0,
        ),
    )

    retrieval = build_retrieval_result(
        observations=observations,
        provenance=provenance,
    )

    with pytest.raises(
        ValueError,
        match="requires exactly two observations",
    ):
        build_computed_answer(
            retrieval,
            operation="difference",
        )


def test_build_computed_answer_calculates_ratio() -> None:
    metadata = DatasetMetadata(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="1.0",
        data_updated_at="2026-08-14T23:00:00+0200",
    )

    provenance = build_retrieval_provenance(
        dataset_code="DEMO_PJAN",
        filters={
            "freq": "A",
            "age": "Y20",
            "sex": "F",
            "geo": "BE",
        },
        metadata=metadata,
        retrieved_at=datetime(
            2026,
            9,
            10,
            12,
            30,
            tzinfo=UTC,
        ),
    )

    observations = (
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
            value=150.0,
        ),
        Observation(
            dataset_code="DEMO_PJAN",
            dimensions={
                "freq": "A",
                "unit": "NR",
                "age": "Y20",
                "sex": "F",
                "geo": "BE",
            },
            time_period="2023",
            value=100.0,
        ),
    )

    retrieval = build_retrieval_result(
        observations=observations,
        provenance=provenance,
    )

    answer = build_computed_answer(
        retrieval,
        operation="ratio",
    )

    assert answer.value == 1.5
    assert answer.unit == "ratio"
    assert answer.computation.operation == "ratio"
    assert answer.computation.input_values == (150.0, 100.0)
    assert answer.computation.output_value == 1.5
    assert answer.provenance == provenance


def test_build_computed_answer_ratio_rejects_zero_denominator() -> None:
    metadata = DatasetMetadata(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="1.0",
        data_updated_at="2026-08-14T23:00:00+0200",
    )

    provenance = build_retrieval_provenance(
        dataset_code="DEMO_PJAN",
        filters={
            "freq": "A",
            "age": "Y20",
            "sex": "F",
            "geo": "BE",
        },
        metadata=metadata,
        retrieved_at=datetime(
            2026,
            9,
            10,
            12,
            30,
            tzinfo=UTC,
        ),
    )

    observations = (
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
            value=150.0,
        ),
        Observation(
            dataset_code="DEMO_PJAN",
            dimensions={
                "freq": "A",
                "unit": "NR",
                "age": "Y20",
                "sex": "F",
                "geo": "BE",
            },
            time_period="2023",
            value=0.0,
        ),
    )

    retrieval = build_retrieval_result(
        observations=observations,
        provenance=provenance,
    )

    with pytest.raises(
        ValueError,
        match="denominator must not be zero",
    ):
        build_computed_answer(
            retrieval,
            operation="ratio",
        )


def test_build_computed_answer_calculates_percentage_change() -> None:
    metadata = DatasetMetadata(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="1.0",
        data_updated_at="2026-08-14T23:00:00+0200",
    )

    provenance = build_retrieval_provenance(
        dataset_code="DEMO_PJAN",
        filters={
            "freq": "A",
            "age": "Y20",
            "sex": "F",
            "geo": "BE",
        },
        metadata=metadata,
        retrieved_at=datetime(
            2026,
            9,
            10,
            12,
            30,
            tzinfo=UTC,
        ),
    )

    observations = (
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
            value=150.0,
        ),
        Observation(
            dataset_code="DEMO_PJAN",
            dimensions={
                "freq": "A",
                "unit": "NR",
                "age": "Y20",
                "sex": "F",
                "geo": "BE",
            },
            time_period="2023",
            value=100.0,
        ),
    )

    retrieval = build_retrieval_result(
        observations=observations,
        provenance=provenance,
    )

    answer = build_computed_answer(
        retrieval,
        operation="percentage_change",
    )

    assert answer.value == 50.0
    assert answer.unit == "percent"
    assert answer.computation.operation == "percentage_change"
    assert answer.computation.input_values == (150.0, 100.0)
    assert answer.computation.output_value == 50.0
    assert answer.provenance == provenance


def test_build_computed_answer_percentage_change_rejects_zero_baseline() -> None:
    metadata = DatasetMetadata(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="1.0",
        data_updated_at="2026-08-14T23:00:00+0200",
    )

    provenance = build_retrieval_provenance(
        dataset_code="DEMO_PJAN",
        filters={
            "freq": "A",
            "age": "Y20",
            "sex": "F",
            "geo": "BE",
        },
        metadata=metadata,
        retrieved_at=datetime(
            2026,
            9,
            10,
            12,
            30,
            tzinfo=UTC,
        ),
    )

    observations = (
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
            value=150.0,
        ),
        Observation(
            dataset_code="DEMO_PJAN",
            dimensions={
                "freq": "A",
                "unit": "NR",
                "age": "Y20",
                "sex": "F",
                "geo": "BE",
            },
            time_period="2023",
            value=0.0,
        ),
    )

    retrieval = build_retrieval_result(
        observations=observations,
        provenance=provenance,
    )

    with pytest.raises(
        ValueError,
        match="baseline must not be zero",
    ):
        build_computed_answer(
            retrieval,
            operation="percentage_change",
        )
