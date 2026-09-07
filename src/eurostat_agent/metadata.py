"""Typed metadata objects for Eurostat SDMX structures.

These objects represent the deterministic metadata extracted from Eurostat.
They deliberately contain no natural-language resolution or LLM logic.

Example:
    >>> ref = CodelistRef(
    ...     agency="ESTAT",
    ...     id="SEX",
    ...     version="2.0",
    ... )
    >>> ref.id
    'SEX'
"""

import re
from dataclasses import dataclass
from xml.etree import ElementTree


@dataclass(frozen=True)
class CodelistRef:
    """Reference to one versioned SDMX codelist."""

    agency: str
    id: str
    version: str


@dataclass(frozen=True)
class Dimension:
    """One dimension in an SDMX data structure."""

    id: str
    position: int | None
    codelist: CodelistRef | None


@dataclass(frozen=True)
class DataStructure:
    """Parsed structural metadata for one Eurostat dataset."""

    id: str
    agency: str
    version: str
    dimensions: tuple[Dimension, ...]
    measure_id: str


@dataclass(frozen=True)
class Code:
    """One code and human-readable label from an SDMX codelist."""

    id: str
    label: str


@dataclass(frozen=True)
class Codelist:
    """One versioned SDMX codelist and its codes."""

    id: str
    agency: str
    version: str
    codes: tuple[Code, ...]


_CODELIST_URN_PATTERN = re.compile(r"Codelist=([^:]+):([^(]+)\(([^)]+)\)")


def _local_name(tag: str) -> str:
    """Return an XML element name without its namespace."""
    return tag.rsplit("}", maxsplit=1)[-1]


def _require_attribute(
    element: ElementTree.Element,
    attribute: str,
) -> str:
    """Return a required XML attribute or raise a clear error."""
    value = element.get(attribute)

    if value is None:
        raise ValueError(
            f"Missing required attribute {attribute!r} on {_local_name(element.tag)!r}."
        )

    return value


def _find_first(
    element: ElementTree.Element,
    name: str,
) -> ElementTree.Element | None:
    """Find the first descendant with the requested local XML name."""
    return next(
        (
            descendant
            for descendant in element.iter()
            if _local_name(descendant.tag) == name
        ),
        None,
    )


def _parse_codelist_ref(
    dimension: ElementTree.Element,
) -> CodelistRef | None:
    """Extract a versioned codelist reference from one dimension."""
    enumeration = _find_first(dimension, "Enumeration")

    if enumeration is None or enumeration.text is None:
        return None

    match = _CODELIST_URN_PATTERN.search(enumeration.text.strip())

    if match is None:
        return None

    agency, codelist_id, version = match.groups()

    return CodelistRef(
        agency=agency,
        id=codelist_id,
        version=version,
    )


def parse_data_structure(xml: str) -> DataStructure:
    """Parse one Eurostat SDMX DataStructure from XML."""
    root = ElementTree.fromstring(xml)

    dsd = _find_first(root, "DataStructure")

    if dsd is None:
        raise ValueError("SDMX response does not contain a DataStructure.")

    dimensions: list[Dimension] = []

    for element in dsd.iter():
        element_name = _local_name(element.tag)

        if element_name not in {"Dimension", "TimeDimension"}:
            continue

        position_text = element.get("position")

        dimensions.append(
            Dimension(
                id=_require_attribute(element, "id"),
                position=(int(position_text) if position_text is not None else None),
                codelist=_parse_codelist_ref(element),
            )
        )

    measure = _find_first(dsd, "Measure")

    if measure is None:
        raise ValueError("DataStructure does not contain a Measure.")

    return DataStructure(
        id=_require_attribute(dsd, "id"),
        agency=_require_attribute(dsd, "agencyID"),
        version=_require_attribute(dsd, "version"),
        dimensions=tuple(dimensions),
        measure_id=_require_attribute(measure, "id"),
    )


def parse_codelist(xml: str) -> Codelist:
    """Parse one Eurostat SDMX Codelist from XML."""
    root = ElementTree.fromstring(xml)

    codelist = _find_first(root, "Codelist")

    if codelist is None:
        raise ValueError("SDMX response does not contain a Codelist.")

    codes: list[Code] = []

    for element in codelist.iter():
        if _local_name(element.tag) != "Code":
            continue

        english_name = next(
            (
                child
                for child in element
                if (
                    _local_name(child.tag) == "Name"
                    and child.get("{http://www.w3.org/XML/1998/namespace}lang") == "en"
                )
            ),
            None,
        )

        if english_name is None or english_name.text is None:
            continue

        codes.append(
            Code(
                id=_require_attribute(element, "id"),
                label=english_name.text,
            )
        )

    return Codelist(
        id=_require_attribute(codelist, "id"),
        agency=_require_attribute(codelist, "agencyID"),
        version=_require_attribute(codelist, "version"),
        codes=tuple(codes),
    )
