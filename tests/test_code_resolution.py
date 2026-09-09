from pathlib import Path

import pytest

from eurostat_agent.metadata import (
    Code,
    Codelist,
    CodelistRef,
    DataStructure,
    Dimension,
    parse_codelist,
    parse_data_structure,
)
from eurostat_agent.resolution import (
    resolve_code,
    resolve_dataset_dimension_code,
    resolve_dimension_code,
)

FIXTURES = Path(__file__).parent / "fixtures" / "sdmx"


def test_resolve_code_matches_exact_official_code() -> None:
    codelist = Codelist(
        id="SEX",
        agency="ESTAT",
        version="2.0",
        codes=(
            Code(id="T", label="Total"),
            Code(id="M", label="Males"),
            Code(id="F", label="Females"),
            Code(id="UNK", label="Unknown"),
        ),
    )

    result = resolve_code(codelist, "F")

    assert result.code == Code(id="F", label="Females")
    assert result.match_type == "exact_code"


def test_resolve_code_matches_exact_label() -> None:
    codelist = Codelist(
        id="SEX",
        agency="ESTAT",
        version="2.0",
        codes=(
            Code(id="T", label="Total"),
            Code(id="M", label="Males"),
            Code(id="F", label="Females"),
            Code(id="UNK", label="Unknown"),
        ),
    )

    result = resolve_code(codelist, "Females")

    assert result.code == Code(id="F", label="Females")
    assert result.match_type == "exact_label"


def test_resolve_code_matches_normalized_label() -> None:
    codelist = Codelist(
        id="SEX",
        agency="ESTAT",
        version="2.0",
        codes=(
            Code(id="T", label="Total"),
            Code(id="M", label="Males"),
            Code(id="F", label="Females"),
            Code(id="UNK", label="Unknown"),
        ),
    )

    result = resolve_code(codelist, "females")

    assert result.code == Code(id="F", label="Females")
    assert result.match_type == "normalized_label"


def test_resolve_code_rejects_ambiguous_exact_label() -> None:
    codelist = Codelist(
        id="EXAMPLE",
        agency="ESTAT",
        version="1.0",
        codes=(
            Code(id="A", label="Total"),
            Code(id="B", label="Total"),
        ),
    )

    with pytest.raises(ValueError, match="Ambiguous"):
        resolve_code(codelist, "Total")


def test_resolve_code_rejects_ambiguous_normalized_label() -> None:
    codelist = Codelist(
        id="EXAMPLE",
        agency="ESTAT",
        version="1.0",
        codes=(
            Code(id="A", label="Females"),
            Code(id="B", label=" FEMALES "),
        ),
    )

    with pytest.raises(ValueError, match="Ambiguous"):
        resolve_code(codelist, "females")


def test_resolve_code_rejects_unknown_value() -> None:
    codelist = Codelist(
        id="SEX",
        agency="ESTAT",
        version="2.0",
        codes=(
            Code(id="T", label="Total"),
            Code(id="M", label="Males"),
            Code(id="F", label="Females"),
            Code(id="UNK", label="Unknown"),
        ),
    )

    with pytest.raises(ValueError, match="No code matches"):
        resolve_code(codelist, "Not a real category")


def test_resolve_dimension_code_uses_dataset_dimension() -> None:
    structure = DataStructure(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="72.0",
        dimensions=(
            Dimension(
                id="sex",
                position=4,
                codelist=CodelistRef(
                    agency="ESTAT",
                    id="SEX",
                    version="2.0",
                ),
            ),
        ),
        measure_id="OBS_VALUE",
    )

    codelist = Codelist(
        id="SEX",
        agency="ESTAT",
        version="2.0",
        codes=(
            Code(id="T", label="Total"),
            Code(id="M", label="Males"),
            Code(id="F", label="Females"),
        ),
    )

    result = resolve_dimension_code(
        structure,
        "sex",
        codelist,
        "Females",
    )

    assert result.code == Code(id="F", label="Females")
    assert result.match_type == "exact_label"


def test_resolve_dimension_code_rejects_unknown_dimension() -> None:
    structure = DataStructure(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="72.0",
        dimensions=(
            Dimension(
                id="sex",
                position=4,
                codelist=CodelistRef(
                    agency="ESTAT",
                    id="SEX",
                    version="2.0",
                ),
            ),
        ),
        measure_id="OBS_VALUE",
    )

    codelist = Codelist(
        id="SEX",
        agency="ESTAT",
        version="2.0",
        codes=(
            Code(id="T", label="Total"),
            Code(id="M", label="Males"),
            Code(id="F", label="Females"),
        ),
    )

    with pytest.raises(ValueError, match="Unknown dimension"):
        resolve_dimension_code(
            structure,
            "geo",
            codelist,
            "Belgium",
        )


def test_resolve_dimension_code_rejects_dimension_without_codelist() -> None:
    structure = DataStructure(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="72.0",
        dimensions=(
            Dimension(
                id="TIME_PERIOD",
                position=6,
                codelist=None,
            ),
        ),
        measure_id="OBS_VALUE",
    )

    codelist = Codelist(
        id="SEX",
        agency="ESTAT",
        version="2.0",
        codes=(Code(id="F", label="Females"),),
    )

    with pytest.raises(ValueError, match="does not reference a codelist"):
        resolve_dimension_code(
            structure,
            "TIME_PERIOD",
            codelist,
            "2024",
        )


def test_resolve_dimension_code_rejects_wrong_codelist() -> None:
    structure = DataStructure(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="72.0",
        dimensions=(
            Dimension(
                id="sex",
                position=4,
                codelist=CodelistRef(
                    agency="ESTAT",
                    id="SEX",
                    version="2.0",
                ),
            ),
        ),
        measure_id="OBS_VALUE",
    )

    codelist = Codelist(
        id="GEO",
        agency="ESTAT",
        version="28.0",
        codes=(Code(id="BE", label="Belgium"),),
    )

    with pytest.raises(ValueError, match="does not match"):
        resolve_dimension_code(
            structure,
            "sex",
            codelist,
            "Belgium",
        )


def test_resolve_dataset_dimension_code_retrieves_dimension_codelist() -> None:
    structure = DataStructure(
        id="DEMO_PJAN",
        agency="ESTAT",
        version="72.0",
        dimensions=(
            Dimension(
                id="sex",
                position=4,
                codelist=CodelistRef(
                    agency="ESTAT",
                    id="SEX",
                    version="2.0",
                ),
            ),
        ),
        measure_id="OBS_VALUE",
    )

    codelist = Codelist(
        id="SEX",
        agency="ESTAT",
        version="2.0",
        codes=(
            Code(id="T", label="Total"),
            Code(id="M", label="Males"),
            Code(id="F", label="Females"),
        ),
    )

    class FakeClient:
        def get_structure(self, dataset_code: str) -> DataStructure:
            assert dataset_code == "DEMO_PJAN"
            return structure

        def get_codelist(self, ref: CodelistRef) -> Codelist:
            assert ref == CodelistRef("ESTAT", "SEX", "2.0")
            return codelist

    result = resolve_dataset_dimension_code(
        FakeClient(),
        "DEMO_PJAN",
        "sex",
        "Females",
    )

    assert result.code == Code(id="F", label="Females")
    assert result.match_type == "exact_label"


def test_resolve_dataset_dimension_code_with_demo_pjan_metadata() -> None:
    structure_xml = (FIXTURES / "demo_pjan_structure.xml").read_text(encoding="utf-8")
    codelist_xml = (FIXTURES / "sex_codelist.xml").read_text(encoding="utf-8")

    structure = parse_data_structure(structure_xml)
    codelist = parse_codelist(codelist_xml)

    class FakeClient:
        def get_structure(self, dataset_code: str) -> DataStructure:
            assert dataset_code == "DEMO_PJAN"
            return structure

        def get_codelist(self, ref: CodelistRef) -> Codelist:
            assert ref == CodelistRef("ESTAT", "SEX", "2.0")
            return codelist

    result = resolve_dataset_dimension_code(
        FakeClient(),
        "DEMO_PJAN",
        "sex",
        "Females",
    )

    assert result.code == Code(id="F", label="Females")
    assert result.match_type == "exact_label"


def test_resolve_code_matches_singular_label_variant() -> None:
    codelist = Codelist(
        id="SEX",
        agency="ESTAT",
        version="2.0",
        codes=(
            Code(id="T", label="Total"),
            Code(id="M", label="Males"),
            Code(id="F", label="Females"),
            Code(id="UNK", label="Unknown"),
        ),
    )

    result = resolve_code(codelist, "female")

    assert result.code == Code(id="F", label="Females")
    assert result.match_type == "label_variant"


def test_resolve_code_matches_numeric_age_label_variant() -> None:
    codelist = Codelist(
        id="AGE",
        agency="ESTAT",
        version="16.0",
        codes=(
            Code(id="TOTAL", label="Total"),
            Code(id="Y20", label="20 years"),
            Code(id="Y_GE100", label="100 years or over"),
        ),
    )

    result = resolve_code(codelist, "20")

    assert result.code == Code(id="Y20", label="20 years")
    assert result.match_type == "label_variant"


def test_resolve_code_does_not_guess_open_ended_age_range() -> None:
    codelist = Codelist(
        id="AGE",
        agency="ESTAT",
        version="16.0",
        codes=(
            Code(id="Y20", label="20 years"),
            Code(id="Y_GE100", label="100 years or over"),
        ),
    )

    with pytest.raises(ValueError, match="No code matches"):
        resolve_code(codelist, "100")


def test_resolve_code_does_not_guess_unrelated_s_suffix() -> None:
    codelist = Codelist(
        id="EXAMPLE",
        agency="ESTAT",
        version="1.0",
        codes=(Code(id="X", label="News"),),
    )

    with pytest.raises(ValueError, match="No code matches"):
        resolve_code(codelist, "new")
