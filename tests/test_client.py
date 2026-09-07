from pathlib import Path

import httpx
import pytest

from eurostat_agent.client import EurostatClient, EurostatClientError
from eurostat_agent.metadata import CodelistRef

FIXTURES = Path(__file__).parent / "fixtures" / "sdmx"


def test_get_structure_parses_eurostat_response() -> None:
    xml = (FIXTURES / "demo_pjan_structure.xml").read_text(encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/structure/dataflow/ESTAT/DEMO_PJAN/1.0")
        assert request.url.params["references"] == "children"

        return httpx.Response(
            status_code=200,
            text=xml,
            headers={
                "Content-Type": ("application/vnd.sdmx.structure+xml;version=3.0.0")
            },
        )

    transport = httpx.MockTransport(handler)

    client = EurostatClient(http_client=httpx.Client(transport=transport))

    structure = client.get_structure("DEMO_PJAN")

    assert structure.id == "DEMO_PJAN"
    assert structure.version == "72.0"


def test_get_codelist_parses_eurostat_response() -> None:
    xml = (FIXTURES / "sex_codelist.xml").read_text(encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/structure/codelist/ESTAT/SEX/2.0")

        return httpx.Response(
            status_code=200,
            text=xml,
            headers={
                "Content-Type": ("application/vnd.sdmx.structure+xml;version=3.0.0")
            },
        )

    transport = httpx.MockTransport(handler)

    client = EurostatClient(http_client=httpx.Client(transport=transport))

    codelist = client.get_codelist(
        CodelistRef(
            agency="ESTAT",
            id="SEX",
            version="2.0",
        )
    )

    assert codelist.id == "SEX"
    assert codelist.version == "2.0"


def test_client_raises_clear_error_on_http_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=404,
            text="Not found",
        )

    transport = httpx.MockTransport(handler)

    client = EurostatClient(http_client=httpx.Client(transport=transport))

    with pytest.raises(
        EurostatClientError,
        match="Eurostat request failed",
    ):
        client.get_structure("DOES_NOT_EXIST")


def test_fetch_series_parses_filtered_observations() -> None:
    csv_text = (FIXTURES / "demo_pjan_observation.csv").read_text(encoding="utf-8-sig")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/data/dataflow/ESTAT/DEMO_PJAN/1.0")
        assert request.url.params["c[freq]"] == "A"
        assert request.url.params["c[age]"] == "Y20"
        assert request.url.params["c[sex]"] == "F"
        assert request.url.params["c[geo]"] == "BE"
        assert request.url.params["c[TIME_PERIOD]"] == "2024"
        assert request.url.params["attributes"] == "none"
        assert request.url.params["measures"] == "all"
        assert request.url.params["compress"] == "false"

        return httpx.Response(
            status_code=200,
            text=csv_text,
            headers={"Content-Type": ("application/vnd.sdmx.data+csv;version=2.0.0")},
        )

    transport = httpx.MockTransport(handler)

    client = EurostatClient(http_client=httpx.Client(transport=transport))

    observations = client.fetch_series(
        dataset_code="DEMO_PJAN",
        filters={
            "freq": "A",
            "age": "Y20",
            "sex": "F",
            "geo": "BE",
            "TIME_PERIOD": "2024",
        },
    )

    assert len(observations) == 1
    assert observations[0].value == 65231.0
    assert observations[0].dimensions["unit"] == "NR"


def test_get_structure_rejects_unexpected_content_type() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            text="<html>Not SDMX</html>",
            headers={
                "Content-Type": "text/html",
            },
        )

    transport = httpx.MockTransport(handler)

    client = EurostatClient(http_client=httpx.Client(transport=transport))

    with pytest.raises(
        EurostatClientError,
        match="unexpected Content-Type",
    ):
        client.get_structure("DEMO_PJAN")


def test_client_context_manager_closes_internal_http_client() -> None:
    client = EurostatClient()

    with client:
        assert not client._http_client.is_closed

    assert client._http_client.is_closed


def test_get_structure_rejects_invalid_sdmx_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            text="<Structure></Structure>",
            headers={
                "Content-Type": ("application/vnd.sdmx.structure+xml;version=3.0.0")
            },
        )

    transport = httpx.MockTransport(handler)

    client = EurostatClient(http_client=httpx.Client(transport=transport))

    with pytest.raises(
        EurostatClientError,
        match="invalid SDMX structure response",
    ):
        client.get_structure("DEMO_PJAN")


def test_get_codelist_rejects_invalid_sdmx_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            text="<Structure></Structure>",
            headers={
                "Content-Type": ("application/vnd.sdmx.structure+xml;version=3.0.0")
            },
        )

    transport = httpx.MockTransport(handler)

    client = EurostatClient(http_client=httpx.Client(transport=transport))

    with pytest.raises(
        EurostatClientError,
        match="invalid SDMX codelist response",
    ):
        client.get_codelist(
            CodelistRef(
                agency="ESTAT",
                id="SEX",
                version="2.0",
            )
        )


def test_fetch_series_rejects_invalid_sdmx_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            text=(
                "STRUCTURE,STRUCTURE_ID,freq,TIME_PERIOD,OBS_VALUE\n"
                "dataflow,INVALID,A,2024,\n"
            ),
            headers={"Content-Type": ("application/vnd.sdmx.data+csv;version=2.0.0")},
        )

    transport = httpx.MockTransport(handler)

    client = EurostatClient(http_client=httpx.Client(transport=transport))

    with pytest.raises(
        EurostatClientError,
        match="invalid SDMX data response",
    ):
        client.fetch_series(
            dataset_code="DEMO_PJAN",
            filters={
                "freq": "A",
                "TIME_PERIOD": "2024",
            },
        )
