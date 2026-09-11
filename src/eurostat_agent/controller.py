from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from eurostat_agent.provenance import (
    DeterministicAnswer,
    RetrievalClient,
    RetrievalResult,
    build_computed_answer,
    build_deterministic_answer,
    retrieve_with_provenance,
)
from eurostat_agent.resolution import (
    MetadataClient,
    resolve_dataset_dimension_code,
)


class ControllerClient(MetadataClient, RetrievalClient, Protocol):
    pass


@dataclass(frozen=True)
class StructuredQuestion:
    dataset_code: str
    filters: dict[str, str]
    operation: str


def resolve_question_filters(
    client: MetadataClient,
    question: StructuredQuestion,
) -> dict[str, str]:
    resolved_filters: dict[str, str] = {}

    for dimension_id, text in question.filters.items():
        if dimension_id == "TIME_PERIOD":
            resolved_filters[dimension_id] = text
            continue

        resolution = resolve_dataset_dimension_code(
            client,
            question.dataset_code,
            dimension_id,
            text,
        )
        resolved_filters[dimension_id] = resolution.code.id

    return resolved_filters


def retrieve_question(
    client: ControllerClient,
    question: StructuredQuestion,
    *,
    retrieved_at: datetime,
) -> RetrievalResult:
    resolved_filters = resolve_question_filters(client, question)

    return retrieve_with_provenance(
        client,
        dataset_code=question.dataset_code,
        filters=resolved_filters,
        retrieved_at=retrieved_at,
    )


def execute_question(
    client: ControllerClient,
    question: StructuredQuestion,
    *,
    retrieved_at: datetime,
) -> DeterministicAnswer:
    retrieval = retrieve_question(
        client,
        question,
        retrieved_at=retrieved_at,
    )

    if question.operation == "none":
        return build_deterministic_answer(retrieval)

    return build_computed_answer(
        retrieval,
        operation=question.operation,
    )
