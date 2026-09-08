from pathlib import Path

import httpx
import pytest

from eurostat_agent.catalogue import search_datasets
from eurostat_agent.catalogue_client import (
    CATALOGUE_TOC_URL,
    CatalogueClient,
    CatalogueClientError,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "catalogue"


def test_catalogue_client_fetches_and_parses_snapshot() -> None:
    xml = (FIXTURE_DIR / "catalogue_sample.xml").read_text(encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url) == CATALOGUE_TOC_URL
        assert request.headers["Accept"] == "application/xml"

        return httpx.Response(
            200,
            text=xml,
            headers={
                "Content-Type": "application/xml",
            },
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as http_client:
        client = CatalogueClient(http_client=http_client)

        snapshot = client.get_snapshot()

    assert snapshot.source_url == CATALOGUE_TOC_URL
    assert snapshot.creation_date == "20260907T2100"

    assert {record.code for record in snapshot.records} == {
        "isoc_ci_ac_i",
        "teibs010",
    }


def test_catalogue_client_wraps_http_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            text="Service unavailable",
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as http_client:
        client = CatalogueClient(http_client=http_client)

        with pytest.raises(
            CatalogueClientError,
            match="Failed to fetch Eurostat catalogue.",
        ) as exc_info:
            client.get_snapshot()

    assert isinstance(
        exc_info.value.__cause__,
        httpx.HTTPStatusError,
    )


def test_catalogue_client_rejects_non_xml_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text="not xml",
            headers={
                "Content-Type": "text/html",
            },
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as http_client:
        client = CatalogueClient(http_client=http_client)

        with pytest.raises(
            ValueError,
            match=(
                "Eurostat catalogue response must have application/xml content type."
            ),
        ):
            client.get_snapshot()


def test_catalogue_client_builds_dataset_index() -> None:
    xml = (FIXTURE_DIR / "catalogue_sample.xml").read_text(encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=xml,
            headers={
                "Content-Type": "application/xml",
            },
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as http_client:
        client = CatalogueClient(http_client=http_client)

        index = client.get_index()

    results = index.search("economic sentiment")

    assert len(results) == 1
    assert results[0].record.code == "teibs010"
    assert results[0].match_type == "title_phrase"


def test_catalogue_discovery_pipeline_searches_fetched_catalogue() -> None:
    xml = (FIXTURE_DIR / "catalogue_sample.xml").read_text(encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=xml,
            headers={
                "Content-Type": "application/xml",
            },
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as http_client:
        client = CatalogueClient(http_client=http_client)

        index = client.get_index()

        results = search_datasets(
            "economic sentiment",
            index=index,
            limit=5,
        )

    assert len(results) == 1

    result = results[0]

    assert result.record.code == "teibs010"
    assert result.record.title == "Economic sentiment indicator"
    assert result.record.product_type == "table"
    assert result.match_type == "title_phrase"
    assert result.score == 70
