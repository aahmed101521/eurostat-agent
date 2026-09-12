import json
from datetime import UTC, datetime

import httpx
import pytest

from eurostat_agent.catalogue import DatasetIndex, DatasetRecord
from eurostat_agent.controller import (
    QuestionPlan,
    answer_planned_question,
)
from eurostat_agent.data import Observation
from eurostat_agent.metadata import (
    Codelist,
    CodelistRef,
    DatasetMetadata,
    DataStructure,
)
from eurostat_agent.model import (
    ModelClientError,
    OpenAICompatibleModel,
)
from eurostat_agent.planner import (
    JsonDatasetSelector,
    JsonPlanner,
)


def test_openai_compatible_model_returns_completion_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url == "https://example.test/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"

        payload = request.read().decode()
        assert '"model":"test-model"' in payload
        assert '"content":"Plan this Eurostat question."' in payload

        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"dataset_code":"DEMO_PJAN"}'}}]
            },
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as http_client:
        model = OpenAICompatibleModel(
            base_url="https://example.test/v1",
            api_key="test-key",
            model="test-model",
            http_client=http_client,
        )

        result = model.complete("Plan this Eurostat question.")

    assert result == '{"dataset_code":"DEMO_PJAN"}'


def test_openai_compatible_model_wraps_http_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={"error": {"message": "provider unavailable"}},
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as http_client:
        model = OpenAICompatibleModel(
            base_url="https://example.test/v1",
            api_key="test-key",
            model="test-model",
            http_client=http_client,
        )

        with pytest.raises(
            ModelClientError,
            match="Model request failed",
        ):
            model.complete("Plan this Eurostat question.")


def test_openai_compatible_model_rejects_malformed_success_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [],
            },
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as http_client:
        model = OpenAICompatibleModel(
            base_url="https://example.test/v1",
            api_key="test-key",
            model="test-model",
            http_client=http_client,
        )

        with pytest.raises(
            ModelClientError,
            match="Model response has invalid structure.",
        ):
            model.complete("Hello")


def test_openai_compatible_model_drives_json_planner() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        content = json.dumps(
            {
                "dataset_query": ("Population on 1 January by age and sex"),
                "filters": {
                    "geo": "Belgium",
                    "TIME_PERIOD": "2024",
                },
                "operation": "none",
            }
        )

        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": content,
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as http_client:
        model = OpenAICompatibleModel(
            base_url="https://example.test/v1",
            api_key="test-key",
            model="test-model",
            http_client=http_client,
        )

        planner = JsonPlanner(model)

        result = planner.plan("What was the population in Belgium in 2024?")

    assert result == QuestionPlan(
        dataset_query="Population on 1 January by age and sex",
        filters={
            "geo": "Belgium",
            "TIME_PERIOD": "2024",
        },
        operation="none",
    )


def test_openai_compatible_model_drives_json_dataset_selector() -> None:
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

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read().decode()

        assert "DEMO_PJAN" in body
        assert "MIGR_IMM1CTZ" in body

        content = json.dumps(
            {
                "dataset_code": "DEMO_PJAN",
            }
        )

        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": content,
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as http_client:
        model = OpenAICompatibleModel(
            base_url="https://example.test/v1",
            api_key="test-key",
            model="test-model",
            http_client=http_client,
        )

        selector = JsonDatasetSelector(model)

        result = selector.select_dataset(
            "What was the population in Belgium in 2024?",
            (
                population,
                migration,
            ),
        )

    assert result == "DEMO_PJAN"


def test_openai_compatible_model_drives_safe_controller_path() -> None:
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
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1

        body = request.read().decode()

        if request_count == 1:
            assert "Convert the following Eurostat question" in body

            content = json.dumps(
                {
                    "dataset_query": ("Population on 1 January by age and sex"),
                    "filters": {
                        "TIME_PERIOD": "2024",
                    },
                    "operation": "none",
                }
            )

            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": content,
                            }
                        }
                    ]
                },
            )

        assert request_count == 2
        assert "Choose the best Eurostat dataset" in body
        assert "DEMO_PJAN" in body

        content = json.dumps(
            {
                "dataset_code": "DEMO_PJAN",
            }
        )

        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": content,
                        }
                    }
                ]
            },
        )

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

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as http_client:
        model = OpenAICompatibleModel(
            base_url="https://example.test/v1",
            api_key="test-key",
            model="test-model",
            http_client=http_client,
        )

        planner = JsonPlanner(model)
        selector = JsonDatasetSelector(model)

        answer = answer_planned_question(
            planner,
            selector,
            FakeClient(),
            "What was the population in 2024?",
            index=index,
            retrieved_at=datetime(
                2026,
                9,
                11,
                12,
                0,
                tzinfo=UTC,
            ),
        )

    assert request_count == 2
    assert answer.value == 123.0
    assert answer.unit == "NR"
    assert answer.computation.operation == "none"
    assert answer.provenance.dataset_code == "DEMO_PJAN"
