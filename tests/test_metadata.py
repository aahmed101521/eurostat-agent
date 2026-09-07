from pathlib import Path

from eurostat_agent.metadata import (
    CodelistRef,
    parse_codelist,
    parse_data_structure,
)

FIXTURES = Path(__file__).parent / "fixtures" / "sdmx"


def test_parse_demo_pjan_structure() -> None:
    xml = (FIXTURES / "demo_pjan_structure.xml").read_text(encoding="utf-8")

    structure = parse_data_structure(xml)

    assert structure.id == "DEMO_PJAN"
    assert structure.agency == "ESTAT"
    assert structure.version == "72.0"
    assert structure.measure_id == "OBS_VALUE"

    assert [dimension.id for dimension in structure.dimensions] == [
        "freq",
        "unit",
        "age",
        "sex",
        "geo",
        "TIME_PERIOD",
    ]


def test_parse_demo_pjan_dimension_codelists() -> None:
    xml = (FIXTURES / "demo_pjan_structure.xml").read_text(encoding="utf-8")

    structure = parse_data_structure(xml)

    references = {
        dimension.id: dimension.codelist for dimension in structure.dimensions
    }

    assert references["freq"] == CodelistRef("ESTAT", "FREQ", "3.9")
    assert references["unit"] == CodelistRef("ESTAT", "UNIT", "82.0")
    assert references["age"] == CodelistRef("ESTAT", "AGE", "16.0")
    assert references["sex"] == CodelistRef("ESTAT", "SEX", "2.0")
    assert references["geo"] == CodelistRef("ESTAT", "GEO", "28.0")
    assert references["TIME_PERIOD"] is None


def test_parse_sex_codelist() -> None:
    xml = (FIXTURES / "sex_codelist.xml").read_text(encoding="utf-8")

    codelist = parse_codelist(xml)

    assert codelist.id == "SEX"
    assert codelist.agency == "ESTAT"
    assert codelist.version == "2.0"

    codes = {code.id: code.label for code in codelist.codes}

    assert codes["T"] == "Total"
    assert codes["M"] == "Males"
    assert codes["F"] == "Females"
    assert codes["UNK"] == "Unknown"
