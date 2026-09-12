import json
from typing import Protocol

from eurostat_agent.catalogue import DatasetRecord
from eurostat_agent.controller import QuestionPlan


class CompletionModel(Protocol):
    def complete(self, prompt: str) -> str: ...


class JsonPlanner:
    def __init__(self, model: CompletionModel) -> None:
        self._model = model

    def plan(self, question: str) -> QuestionPlan:
        prompt = (
            "Convert the following Eurostat question into JSON with keys "
            "'dataset_query', 'filters', and 'operation'.\n\n"
            f"Question: {question}"
        )

        raw = self._model.complete(prompt)

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Planner returned invalid JSON.") from exc

        if not isinstance(payload, dict):
            raise ValueError("Planner response must be a JSON object.")

        try:
            dataset_query = payload["dataset_query"]
            filters = payload["filters"]
            operation = payload["operation"]
        except KeyError as exc:
            raise ValueError("Planner response is missing required fields.") from exc

        expected_fields = {
            "dataset_query",
            "filters",
            "operation",
        }

        if set(payload) != expected_fields:
            raise ValueError("Planner response contains unexpected fields.")

        if (
            not isinstance(dataset_query, str)
            or not isinstance(filters, dict)
            or not isinstance(operation, str)
            or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in filters.items()
            )
        ):
            raise ValueError("Planner response has invalid field types.")

        return QuestionPlan(
            dataset_query=dataset_query,
            filters=filters,
            operation=operation,
        )


class JsonDatasetSelector:
    def __init__(self, model: CompletionModel) -> None:
        self._model = model

    def select_dataset(
        self,
        question: str,
        candidates: tuple[DatasetRecord, ...],
    ) -> str:
        candidate_text = "\n".join(
            f"- {candidate.code}: {candidate.title}" for candidate in candidates
        )

        prompt = (
            "Choose the best Eurostat dataset for the question from the "
            "candidate datasets below.\n\n"
            f"Question: {question}\n\n"
            f"Candidates:\n{candidate_text}\n\n"
            "Return JSON with the key 'dataset_code'."
        )

        raw = self._model.complete(prompt)

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Dataset selector returned invalid JSON.") from exc

        if not isinstance(payload, dict):
            raise ValueError("Dataset selector response must be a JSON object.")

        try:
            dataset_code = payload["dataset_code"]
        except KeyError as exc:
            raise ValueError(
                "Dataset selector response is missing required fields."
            ) from exc

        if set(payload) != {"dataset_code"}:
            raise ValueError("Dataset selector response contains unexpected fields.")

        if not isinstance(dataset_code, str):
            raise ValueError("Dataset selector response has invalid field types.")

        return dataset_code
