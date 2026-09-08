from __future__ import annotations

import httpx

from eurostat_agent.catalogue import (
    CatalogueSnapshot,
    DatasetIndex,
    parse_catalogue_snapshot,
)

CATALOGUE_TOC_URL = "https://ec.europa.eu/eurostat/api/dissemination/catalogue/toc/xml"


class CatalogueClientError(RuntimeError):
    """Raised when the Eurostat catalogue cannot be retrieved."""


class CatalogueClient:
    def __init__(
        self,
        *,
        http_client: httpx.Client,
    ) -> None:
        self._http_client = http_client

    def get_snapshot(self) -> CatalogueSnapshot:
        try:
            response = self._http_client.get(
                CATALOGUE_TOC_URL,
                headers={
                    "Accept": "application/xml",
                },
            )

            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise CatalogueClientError("Failed to fetch Eurostat catalogue.") from exc

        content_type = response.headers.get("Content-Type", "")

        if "application/xml" not in content_type.casefold():
            raise ValueError(
                "Eurostat catalogue response must have application/xml content type."
            )

        return parse_catalogue_snapshot(
            response.text,
            source_url=CATALOGUE_TOC_URL,
        )

    def get_index(self) -> DatasetIndex:
        snapshot = self.get_snapshot()

        return DatasetIndex(snapshot.records)
