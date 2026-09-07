"""HTTP client for deterministic Eurostat SDMX retrieval.

The client is responsible only for transport and response validation.
It does not perform natural-language resolution or agent reasoning.

Example:
    >>> client = EurostatClient()
    >>> structure = client.get_structure("DEMO_PJAN")
    >>> structure.id
    'DEMO_PJAN'
"""

from __future__ import annotations

import httpx

from eurostat_agent.data import Observation, parse_sdmx_csv
from eurostat_agent.metadata import (
    Codelist,
    CodelistRef,
    DataStructure,
    parse_codelist,
    parse_data_structure,
)


class EurostatClientError(RuntimeError):
    """Raised when a Eurostat request cannot be completed successfully."""


class EurostatClient:
    """Deterministic HTTP client for Eurostat SDMX retrieval."""

    BASE_URL = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0"

    def __init__(
        self,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        """Create a Eurostat client.

        Args:
            http_client: Optional HTTPX client, primarily for testing.
        """
        self._owns_http_client = http_client is None

        self._http_client = http_client or httpx.Client(
            timeout=30.0,
            headers={"Accept": ("application/vnd.sdmx.structure+xml;version=3.0.0")},
        )

    def close(self) -> None:
        """Close HTTP resources owned by this client."""
        if self._owns_http_client:
            self._http_client.close()

    def __enter__(self) -> EurostatClient:
        """Enter the client context manager."""
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        """Exit the client context manager and release resources."""
        self.close()

    def get_structure(
        self,
        dataset_code: str,
    ) -> DataStructure:
        """Retrieve and parse the current structure for one dataset."""
        url = f"{self.BASE_URL}/structure/dataflow/ESTAT/{dataset_code}/1.0"

        xml = self._get_text(
            url,
            params={
                "references": "children",
                "compress": "false",
            },
            expected_content_type=("application/vnd.sdmx.structure+xml"),
        )

        try:
            return parse_data_structure(xml)
        except ValueError as exc:
            raise EurostatClientError(
                f"Eurostat returned an invalid SDMX structure response: {exc}"
            ) from exc

    def get_codelist(
        self,
        ref: CodelistRef,
    ) -> Codelist:
        """Retrieve and parse one exact versioned codelist."""
        url = f"{self.BASE_URL}/structure/codelist/{ref.agency}/{ref.id}/{ref.version}"

        xml = self._get_text(
            url,
            params={
                "compress": "false",
            },
            expected_content_type=("application/vnd.sdmx.structure+xml"),
        )

        return parse_codelist(xml)

    def fetch_series(
        self,
        dataset_code: str,
        filters: dict[str, str],
    ) -> tuple[Observation, ...]:
        """Retrieve filtered Eurostat observations as typed objects."""
        url = f"{self.BASE_URL}/data/dataflow/ESTAT/{dataset_code}/1.0"

        params = {f"c[{dimension}]": code for dimension, code in filters.items()}

        params.update(
            {
                "attributes": "none",
                "measures": "all",
                "compress": "false",
            }
        )

        csv_text = self._get_text(
            url,
            params=params,
            headers={
                "Accept": ("application/vnd.sdmx.data+csv;version=2.0.0;labels=id")
            },
            expected_content_type=("application/vnd.sdmx.data+csv"),
        )

        return parse_sdmx_csv(csv_text)

    def _get_text(
        self,
        url: str,
        *,
        params: dict[str, str],
        headers: dict[str, str] | None = None,
        expected_content_type: str | None = None,
    ) -> str:
        """Perform one validated HTTP GET request."""
        try:
            response = self._http_client.get(
                url,
                params=params,
                headers=headers,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise EurostatClientError(f"Eurostat request failed: {exc}") from exc

        if expected_content_type is not None:
            content_type = response.headers.get(
                "Content-Type",
                "",
            )

            if expected_content_type not in content_type:
                raise EurostatClientError(
                    "Eurostat returned unexpected Content-Type: "
                    f"{content_type!r}; expected "
                    f"{expected_content_type!r}."
                )

        return response.text
