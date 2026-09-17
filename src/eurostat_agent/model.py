import json
from collections.abc import Mapping

import httpx


class ModelClientError(RuntimeError):
    """Raised when a model request cannot be completed successfully."""


class OpenAICompatibleModel:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        http_client: httpx.Client,
        request_options: Mapping[str, object] | None = None,
    ) -> None:
        options = dict(request_options or {})

        reserved_fields = {
            "model",
            "messages",
        }

        if reserved_fields.intersection(options):
            raise ValueError(
                "Model request options must not override model or messages."
            )

        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._http_client = http_client
        self._request_options = options

    def complete(self, prompt: str) -> str:
        payload: dict[str, object] = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }

        payload.update(self._request_options)

        try:
            response = self._http_client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                },
                json=payload,
            )

            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ModelClientError("Model request failed.") from exc

        try:
            response_payload = response.json()
        except json.JSONDecodeError as exc:
            raise ModelClientError("Model response has invalid structure.") from exc

        try:
            content = response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ModelClientError("Model response has invalid structure.") from exc

        if not isinstance(content, str):
            raise ModelClientError("Model response has invalid structure.")

        return content
