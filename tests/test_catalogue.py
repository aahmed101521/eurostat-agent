from pathlib import Path

from eurostat_agent.catalogue import (
    CatalogueSnapshot,
    DatasetRecord,
    parse_catalogue,
    parse_catalogue_snapshot,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "catalogue"


def test_parse_catalogue_deduplicates_codes_and_preserves_paths() -> None:
    xml = (FIXTURE_DIR / "catalogue_sample.xml").read_text(encoding="utf-8")

    records = parse_catalogue(xml)

    assert len(records) == 2

    by_code = {record.code: record for record in records}

    assert by_code["isoc_ci_ac_i"] == DatasetRecord(
        code="isoc_ci_ac_i",
        title="Individuals - internet activities",
        product_type="dataset",
        description="Internet activities performed by individuals.",
        last_update="28.08.2026",
        last_modified="28.08.2026",
        data_start="2010",
        data_end="2026",
        value_count=123456,
        paths=(
            (
                "Database by themes",
                "Science, technology, digital society",
            ),
            (
                "Cross cutting topics",
                "Children and youth",
                "Youth",
            ),
        ),
    )

    assert by_code["teibs010"] == DatasetRecord(
        code="teibs010",
        title="Economic sentiment indicator",
        product_type="table",
        description="Composite indicator tracking economic sentiment.",
        last_update="28.08.2026",
        last_modified="28.08.2026",
        data_start="2025-09",
        data_end="2026-08",
        value_count=404,
        paths=(
            (
                "Database by themes",
                "Science, technology, digital society",
            ),
        ),
    )


def test_parse_catalogue_finds_leaves_nested_inside_dataset_leaves() -> None:
    xml = (FIXTURE_DIR / "catalogue_nested_leaf.xml").read_text(encoding="utf-8")

    records = parse_catalogue(xml)

    assert {record.code for record in records} == {
        "ei_bssi_m_r2",
        "teibs010",
    }

    by_code = {record.code: record for record in records}

    assert by_code["teibs010"].product_type == "table"

    assert by_code["teibs010"].paths == (
        (
            "Database by themes",
            "Economic sentiment and confidence indicators by sector",
        ),
    )


def test_parse_catalogue_snapshot_preserves_source_provenance() -> None:
    xml = (FIXTURE_DIR / "catalogue_sample.xml").read_text(encoding="utf-8")

    snapshot = parse_catalogue_snapshot(
        xml,
        source_url=(
            "https://ec.europa.eu/eurostat/api/dissemination/catalogue/toc/xml"
        ),
    )

    assert isinstance(
        snapshot,
        CatalogueSnapshot,
    )

    assert snapshot.creation_date == "20260907T2100"

    assert snapshot.source_url == (
        "https://ec.europa.eu/eurostat/api/dissemination/catalogue/toc/xml"
    )

    assert len(snapshot.records) == 2

    assert {record.code for record in snapshot.records} == {
        "isoc_ci_ac_i",
        "teibs010",
    }
