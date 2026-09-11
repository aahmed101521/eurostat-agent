import json
from typing import Protocol

from eurostat_agent.controller import StructuredQuestion


class CompletionModel(Protocol):
    def complete(self, prompt: str) -> str: ...


class JsonPlanner:
    def __init__(self, model: CompletionModel) -> None:
        self._model = model

    def plan(self, question: str) -> StructuredQuestion:
        prompt = (
            "Convert the following Eurostat question into JSON with keys "
            "'dataset_code', 'filters', and 'operation'.\n\n"
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
            dataset_code = payload["dataset_code"]
            filters = payload["filters"]
            operation = payload["operation"]
        except KeyError as exc:
            raise ValueError("Planner response is missing required fields.") from exc

        expected_fields = {
            "dataset_code",
            "filters",
            "operation",
        }

        if set(payload) != expected_fields:
            raise ValueError("Planner response contains unexpected fields.")

        if (
            not isinstance(dataset_code, str)
            or not isinstance(filters, dict)
            or not isinstance(operation, str)
            or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in filters.items()
            )
        ):
            raise ValueError("Planner response has invalid field types.")

        return StructuredQuestion(
            dataset_code=dataset_code,
            filters=filters,
            operation=operation,
        )
