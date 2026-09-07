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
