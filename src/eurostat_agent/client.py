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
    """Deterministic HTTP client for Eurostat SDMX metadata."""

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
        self._http_client = http_client or httpx.Client(
            timeout=30.0,
            headers={"Accept": ("application/vnd.sdmx.structure+xml;version=3.0.0")},
        )

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
        )

        return parse_data_structure(xml)

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
        )

        return parse_codelist(xml)

    def _get_text(
        self,
        url: str,
        *,
        params: dict[str, str],
    ) -> str:
        """Perform one validated HTTP GET request."""
        try:
            response = self._http_client.get(
                url,
                params=params,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise EurostatClientError(f"Eurostat request failed: {exc}") from exc

        return response.text
