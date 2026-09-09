"""Deterministic natural-language resolution of Eurostat codelist values."""

from dataclasses import dataclass
from typing import Protocol

from eurostat_agent.metadata import (
    Code,
    Codelist,
    CodelistRef,
    DataStructure,
)

_LABEL_VARIANTS = {
    "female": "females",
    "male": "males",
}


@dataclass(frozen=True)
class CodeResolution:
    """One deterministic resolution to an official Eurostat code."""

    code: Code
    match_type: str


class MetadataClient(Protocol):
    """Minimal metadata retrieval interface needed for code resolution."""

    def get_structure(
        self,
        dataset_code: str,
    ) -> DataStructure:
        """Retrieve the structure for one dataset."""
        ...

    def get_codelist(
        self,
        ref: CodelistRef,
    ) -> Codelist:
        """Retrieve one versioned codelist."""
        ...


def _normalize_text(text: str) -> str:
    """Normalize text for safe deterministic comparison."""
    return text.strip().casefold()


def _is_label_variant(
    label: str,
    text: str,
) -> bool:
    """Return whether a label is a controlled deterministic text variant."""
    normalized_label = _normalize_text(label)
    normalized_text = _normalize_text(text)

    known_variant = _LABEL_VARIANTS.get(normalized_text)

    if known_variant is not None and normalized_label == known_variant:
        return True

    if normalized_text.isdigit():
        return normalized_label in {
            f"{normalized_text} year",
            f"{normalized_text} years",
        }

    return False


def resolve_code(
    codelist: Codelist,
    text: str,
) -> CodeResolution:
    """Resolve text against an official Eurostat codelist."""
    for code in codelist.codes:
        if code.id == text:
            return CodeResolution(
                code=code,
                match_type="exact_code",
            )

    exact_label_matches = tuple(code for code in codelist.codes if code.label == text)

    if len(exact_label_matches) > 1:
        raise ValueError(f"Ambiguous label {text!r} in codelist {codelist.id!r}.")

    if len(exact_label_matches) == 1:
        return CodeResolution(
            code=exact_label_matches[0],
            match_type="exact_label",
        )

    normalized_text = _normalize_text(text)

    normalized_label_matches = tuple(
        code
        for code in codelist.codes
        if _normalize_text(code.label) == normalized_text
    )

    if len(normalized_label_matches) > 1:
        raise ValueError(
            f"Ambiguous normalized label {text!r} in codelist {codelist.id!r}."
        )

    if len(normalized_label_matches) == 1:
        return CodeResolution(
            code=normalized_label_matches[0],
            match_type="normalized_label",
        )

    label_variant_matches = tuple(
        code for code in codelist.codes if _is_label_variant(code.label, text)
    )

    if len(label_variant_matches) > 1:
        raise ValueError(
            f"Ambiguous label variant {text!r} in codelist {codelist.id!r}."
        )

    if len(label_variant_matches) == 1:
        return CodeResolution(
            code=label_variant_matches[0],
            match_type="label_variant",
        )

    raise ValueError(f"No code matches {text!r} in codelist {codelist.id!r}.")


def resolve_dimension_code(
    structure: DataStructure,
    dimension_id: str,
    codelist: Codelist,
    text: str,
) -> CodeResolution:
    """Resolve text for one dimension of a dataset structure."""
    dimension = next(
        (
            dimension
            for dimension in structure.dimensions
            if dimension.id == dimension_id
        ),
        None,
    )

    if dimension is None:
        raise ValueError(
            f"Unknown dimension {dimension_id!r} for data structure {structure.id!r}."
        )

    if dimension.codelist is None:
        raise ValueError(f"Dimension {dimension_id!r} does not reference a codelist.")

    expected = dimension.codelist

    if (
        codelist.agency != expected.agency
        or codelist.id != expected.id
        or codelist.version != expected.version
    ):
        raise ValueError(
            f"Codelist {codelist.agency!r}/{codelist.id!r}/{codelist.version!r}"
            f" does not match dimension {dimension_id!r} codelist "
            f"{expected.agency!r}/{expected.id!r}/{expected.version!r}."
        )

    return resolve_code(codelist, text)


def resolve_dataset_dimension_code(
    client: MetadataClient,
    dataset_code: str,
    dimension_id: str,
    text: str,
) -> CodeResolution:
    """Resolve text using metadata retrieved for one dataset dimension."""
    structure = client.get_structure(dataset_code)

    dimension = next(
        (
            dimension
            for dimension in structure.dimensions
            if dimension.id == dimension_id
        ),
        None,
    )

    if dimension is None:
        raise ValueError(
            f"Unknown dimension {dimension_id!r} for data structure {structure.id!r}."
        )

    if dimension.codelist is None:
        raise ValueError(f"Dimension {dimension_id!r} does not reference a codelist.")

    codelist = client.get_codelist(dimension.codelist)

    return resolve_dimension_code(
        structure,
        dimension_id,
        codelist,
        text,
    )
