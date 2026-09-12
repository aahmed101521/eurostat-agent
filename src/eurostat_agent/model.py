from typing import cast

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
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._http_client = http_client

    def complete(self, prompt: str) -> str:
        try:
            response = self._http_client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                },
                json={
                    "model": self._model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                },
            )

            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ModelClientError("Model request failed.") from exc

        payload = response.json()

        return cast(str, payload["choices"][0]["message"]["content"])
