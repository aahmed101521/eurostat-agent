import httpx
import pytest

from eurostat_agent.model import ModelClientError, OpenAICompatibleModel


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
