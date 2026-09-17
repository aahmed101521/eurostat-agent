from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from eurostat_agent.catalogue import (
    DatasetIndex,
    DatasetRecord,
    DatasetSearchResult,
    search_datasets,
)
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
from eurostat_agent.tracing import TraceRecorder


class ControllerClient(MetadataClient, RetrievalClient, Protocol):
    pass


@dataclass(frozen=True)
class StructuredQuestion:
    dataset_code: str
    filters: dict[str, str]
    operation: str


@dataclass(frozen=True)
class QuestionPlan:
    dataset_query: str
    filters: dict[str, str]
    operation: str


class QuestionPlanner(Protocol):
    def plan(self, question: str) -> QuestionPlan: ...


class DatasetSelector(Protocol):
    def select_dataset(
        self,
        question: str,
        candidates: tuple[DatasetRecord, ...],
    ) -> str: ...


def create_question_plan(
    planner: QuestionPlanner,
    question: str,
) -> QuestionPlan:
    return planner.plan(question)


def discover_dataset_candidates(
    query: str,
    *,
    index: DatasetIndex,
    limit: int = 10,
) -> tuple[DatasetSearchResult, ...]:
    return search_datasets(
        query,
        index=index,
        limit=limit,
    )


def select_dataset_candidate(
    selector: DatasetSelector,
    question: str,
    candidates: tuple[DatasetRecord, ...],
) -> DatasetRecord:
    selected_code = selector.select_dataset(
        question,
        candidates,
    )

    for candidate in candidates:
        if candidate.code == selected_code:
            return candidate

    raise ValueError(
        f"Selected dataset {selected_code!r} is not among the discovered candidates."
    )


def discover_and_select_dataset(
    selector: DatasetSelector,
    *,
    question: str,
    search_query: str,
    index: DatasetIndex,
    limit: int = 10,
) -> DatasetRecord:
    search_results = discover_dataset_candidates(
        search_query,
        index=index,
        limit=limit,
    )

    candidates = tuple(result.record for result in search_results)

    if not candidates:
        raise ValueError("No dataset candidates were found.")

    return select_dataset_candidate(
        selector,
        question,
        candidates,
    )


def build_structured_question(
    plan: QuestionPlan,
    dataset: DatasetRecord,
) -> StructuredQuestion:
    return StructuredQuestion(
        dataset_code=dataset.code,
        filters=plan.filters,
        operation=plan.operation,
    )


def materialize_question_plan(
    selector: DatasetSelector,
    plan: QuestionPlan,
    *,
    question: str,
    index: DatasetIndex,
    limit: int = 10,
) -> StructuredQuestion:
    dataset = discover_and_select_dataset(
        selector,
        question=question,
        search_query=plan.dataset_query,
        index=index,
        limit=limit,
    )

    return build_structured_question(
        plan,
        dataset,
    )


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
    resolved_filters = resolve_question_filters(
        client,
        question,
    )

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


def answer_planned_question(
    planner: QuestionPlanner,
    selector: DatasetSelector,
    client: ControllerClient,
    question: str,
    *,
    index: DatasetIndex,
    retrieved_at: datetime,
    limit: int = 10,
    trace_recorder: TraceRecorder | None = None,
    planning_metadata: tuple[tuple[str, str], ...] = (),
    selection_metadata: tuple[tuple[str, str], ...] = (),
) -> DeterministicAnswer:
    total_stage = (
        trace_recorder.stage("controller_total")
        if trace_recorder is not None
        else nullcontext()
    )

    with total_stage:
        planning_stage = (
            trace_recorder.stage("planning", metadata=planning_metadata)
            if trace_recorder is not None
            else nullcontext()
        )
        with planning_stage:
            plan = create_question_plan(
                planner,
                question,
            )

        catalogue_stage = (
            trace_recorder.stage("catalogue_search")
            if trace_recorder is not None
            else nullcontext()
        )
        with catalogue_stage:
            search_results = discover_dataset_candidates(
                plan.dataset_query,
                index=index,
                limit=limit,
            )

        candidates = tuple(result.record for result in search_results)
        candidate_metadata = (
            ("candidate_count", str(len(candidates))),
            *selection_metadata,
        )
        selection_stage = (
            trace_recorder.stage(
                "dataset_selection",
                metadata=candidate_metadata,
            )
            if trace_recorder is not None
            else nullcontext()
        )
        with selection_stage:
            if not candidates:
                raise ValueError("No dataset candidates were found.")

            dataset = select_dataset_candidate(
                selector,
                question,
                candidates,
            )

        structured_question = build_structured_question(
            plan,
            dataset,
        )

        resolution_stage = (
            trace_recorder.stage(
                "filter_resolution",
                metadata=(("dataset", structured_question.dataset_code),),
            )
            if trace_recorder is not None
            else nullcontext()
        )
        with resolution_stage:
            resolved_filters = resolve_question_filters(
                client,
                structured_question,
            )

        retrieval_stage = (
            trace_recorder.stage(
                "retrieval",
                metadata=(("dataset", structured_question.dataset_code),),
            )
            if trace_recorder is not None
            else nullcontext()
        )
        with retrieval_stage:
            retrieval = retrieve_with_provenance(
                client,
                dataset_code=structured_question.dataset_code,
                filters=resolved_filters,
                retrieved_at=retrieved_at,
            )

        computation_stage = (
            trace_recorder.stage(
                "computation",
                metadata=(("operation", structured_question.operation),),
            )
            if trace_recorder is not None
            else nullcontext()
        )
        with computation_stage:
            if structured_question.operation == "none":
                return build_deterministic_answer(retrieval)

            return build_computed_answer(
                retrieval,
                operation=structured_question.operation,
            )
