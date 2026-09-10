"""Deterministic provenance records for Eurostat retrievals and answers."""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from eurostat_agent.data import Observation
from eurostat_agent.metadata import DatasetMetadata


@dataclass(frozen=True)
class RetrievalProvenance:
    """Immutable provenance for one deterministic Eurostat retrieval."""

    dataset_code: str
    dataset_agency: str
    dataset_version: str
    filters: tuple[tuple[str, str], ...]
    retrieved_at: datetime
    data_updated_at: str
    source: str
    source_url: str


@dataclass(frozen=True)
class RetrievalResult:
    """One deterministic Eurostat retrieval with its provenance."""

    observations: tuple[Observation, ...]
    provenance: RetrievalProvenance


@dataclass(frozen=True)
class ComputationProvenance:
    """Immutable provenance for one deterministic computation."""

    operation: str
    input_values: tuple[float, ...]
    input_time_periods: tuple[str, ...]
    input_dimensions: tuple[
        tuple[tuple[str, str], ...],
        ...,
    ]
    output_value: float


@dataclass(frozen=True)
class DeterministicAnswer:
    """One answer-ready deterministic statistical result."""

    value: float
    unit: str
    computation: ComputationProvenance
    provenance: RetrievalProvenance


class RetrievalClient(Protocol):
    """Minimal client interface required for provenance-aware retrieval."""

    def get_dataset_metadata(
        self,
        dataset_code: str,
    ) -> DatasetMetadata:
        """Retrieve dataset-level provenance metadata."""
        ...

    def fetch_series(
        self,
        dataset_code: str,
        filters: dict[str, str],
    ) -> tuple[Observation, ...]:
        """Retrieve observations for exact dimension filters."""
        ...


def build_retrieval_provenance(
    *,
    dataset_code: str,
    filters: dict[str, str],
    metadata: DatasetMetadata,
    retrieved_at: datetime,
) -> RetrievalProvenance:
    """Build canonical provenance for one Eurostat retrieval."""
    if metadata.id != dataset_code:
        raise ValueError(
            f"Dataset metadata {metadata.id!r} does not match "
            f"requested dataset {dataset_code!r}."
        )

    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        raise ValueError("Retrieval timestamp must be timezone-aware.")

    source_url = (
        "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/"
        f"data/dataflow/{metadata.agency}/{dataset_code}/{metadata.version}"
    )

    return RetrievalProvenance(
        dataset_code=dataset_code,
        dataset_agency=metadata.agency,
        dataset_version=metadata.version,
        filters=tuple(sorted(filters.items())),
        retrieved_at=retrieved_at,
        data_updated_at=metadata.data_updated_at,
        source="Eurostat SDMX 3.0",
        source_url=source_url,
    )


def _observation_filter_value(
    observation: Observation,
    dimension: str,
) -> str | None:
    """Return an observation value for one provenance filter."""
    if dimension == "TIME_PERIOD":
        return observation.time_period

    return observation.dimensions.get(dimension)


def _canonical_dimensions(
    observation: Observation,
) -> tuple[tuple[str, str], ...]:
    """Return one observation's dimensions in deterministic order."""
    return tuple(sorted(observation.dimensions.items()))


def build_retrieval_result(
    *,
    observations: tuple[Observation, ...],
    provenance: RetrievalProvenance,
) -> RetrievalResult:
    """Bind retrieved observations to validated retrieval provenance."""
    for observation in observations:
        if observation.dataset_code != provenance.dataset_code:
            raise ValueError(
                f"Observation dataset {observation.dataset_code!r} "
                f"does not match provenance dataset "
                f"{provenance.dataset_code!r}."
            )

        for dimension, expected_value in provenance.filters:
            observed_value = _observation_filter_value(
                observation,
                dimension,
            )

            if observed_value != expected_value:
                raise ValueError(
                    f"Observation does not satisfy provenance filter "
                    f"{dimension!r}={expected_value!r}; "
                    f"observed {observed_value!r}."
                )

    return RetrievalResult(
        observations=observations,
        provenance=provenance,
    )


def retrieve_with_provenance(
    client: RetrievalClient,
    *,
    dataset_code: str,
    filters: dict[str, str],
    retrieved_at: datetime,
) -> RetrievalResult:
    """Retrieve Eurostat observations together with validated provenance."""
    metadata = client.get_dataset_metadata(dataset_code)

    observations = client.fetch_series(
        dataset_code,
        filters,
    )

    provenance = build_retrieval_provenance(
        dataset_code=dataset_code,
        filters=filters,
        metadata=metadata,
        retrieved_at=retrieved_at,
    )

    return build_retrieval_result(
        observations=observations,
        provenance=provenance,
    )


def build_deterministic_answer(
    retrieval: RetrievalResult,
) -> DeterministicAnswer:
    """Build an answer from one directly retrieved observation."""
    if len(retrieval.observations) != 1:
        raise ValueError(
            "A direct deterministic answer requires exactly one observation."
        )

    observation = retrieval.observations[0]

    unit = observation.dimensions.get("unit")

    if unit is None:
        raise ValueError("Observation does not contain a unit dimension.")

    computation = ComputationProvenance(
        operation="none",
        input_values=(observation.value,),
        input_time_periods=(observation.time_period,),
        input_dimensions=(_canonical_dimensions(observation),),
        output_value=observation.value,
    )

    return DeterministicAnswer(
        value=observation.value,
        unit=unit,
        computation=computation,
        provenance=retrieval.provenance,
    )


def build_computed_answer(
    retrieval: RetrievalResult,
    *,
    operation: str,
) -> DeterministicAnswer:
    """Build an answer by applying a deterministic computation."""
    if operation not in {
        "sum",
        "difference",
        "ratio",
        "percentage_change",
    }:
        raise ValueError(f"Unsupported computation operation: {operation!r}.")

    if not retrieval.observations:
        raise ValueError("A computed answer requires at least one observation.")

    first_unit = retrieval.observations[0].dimensions.get("unit")

    if first_unit is None:
        raise ValueError("Observation does not contain a unit dimension.")

    for observation in retrieval.observations[1:]:
        unit = observation.dimensions.get("unit")

        if unit is None:
            raise ValueError("Observation does not contain a unit dimension.")

        if unit != first_unit:
            raise ValueError(
                "All observations in a computed answer must use the same unit."
            )

    input_values = tuple(observation.value for observation in retrieval.observations)

    input_time_periods = tuple(
        observation.time_period for observation in retrieval.observations
    )

    input_dimensions = tuple(
        _canonical_dimensions(observation) for observation in retrieval.observations
    )

    if operation == "sum":
        output_value = sum(input_values)
        output_unit = first_unit

    elif operation == "difference":
        if len(input_values) != 2:
            raise ValueError(
                "A difference computation requires exactly two observations."
            )

        output_value = input_values[0] - input_values[1]
        output_unit = first_unit

    elif operation == "ratio":
        if len(input_values) != 2:
            raise ValueError("A ratio computation requires exactly two observations.")

        if input_values[1] == 0:
            raise ValueError("A ratio computation denominator must not be zero.")

        output_value = input_values[0] / input_values[1]
        output_unit = "ratio"

    else:
        if len(input_values) != 2:
            raise ValueError(
                "A percentage change computation requires exactly two observations."
            )

        if input_values[1] == 0:
            raise ValueError(
                "A percentage change computation baseline must not be zero."
            )

        output_value = ((input_values[0] - input_values[1]) / input_values[1]) * 100
        output_unit = "percent"

    computation = ComputationProvenance(
        operation=operation,
        input_values=input_values,
        input_time_periods=input_time_periods,
        input_dimensions=input_dimensions,
        output_value=output_value,
    )

    return DeterministicAnswer(
        value=output_value,
        unit=output_unit,
        computation=computation,
        provenance=retrieval.provenance,
    )
