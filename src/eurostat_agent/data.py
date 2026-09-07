"""Typed observations parsed from Eurostat SDMX-CSV responses.

The data layer represents values returned by Eurostat after the dataset and
dimension filters have already been chosen. It contains no natural-language
resolution or agent reasoning.

Example:
    >>> observation = Observation(
    ...     dataset_code="DEMO_PJAN",
    ...     dimensions={
    ...         "freq": "A",
    ...         "unit": "NR",
    ...         "age": "Y20",
    ...         "sex": "F",
    ...         "geo": "BE",
    ...     },
    ...     time_period="2024",
    ...     value=65231.0,
    ... )
    >>> observation.value
    65231.0
"""

import csv
import re
from dataclasses import dataclass
from io import StringIO


@dataclass(frozen=True)
class Observation:
    """One Eurostat observation with its complete dimension coordinates."""

    dataset_code: str
    dimensions: dict[str, str]
    time_period: str
    value: float


_STRUCTURE_ID_PATTERN = re.compile(r"^[^:]+:([^(]+)\([^)]+\)$")

_NON_DIMENSION_COLUMNS = {
    "STRUCTURE",
    "STRUCTURE_ID",
    "TIME_PERIOD",
    "OBS_VALUE",
}


def _dataset_code_from_structure_id(structure_id: str) -> str:
    """Extract the Eurostat dataset code from an SDMX structure identifier."""
    match = _STRUCTURE_ID_PATTERN.match(structure_id)

    if match is None:
        raise ValueError(f"Invalid SDMX STRUCTURE_ID: {structure_id!r}.")

    return match.group(1)


def parse_sdmx_csv(csv_text: str) -> tuple[Observation, ...]:
    """Parse Eurostat SDMX-CSV observations into typed objects."""
    reader = csv.DictReader(StringIO(csv_text))

    if reader.fieldnames is None:
        raise ValueError("SDMX-CSV response does not contain a header.")

    observations: list[Observation] = []

    for row in reader:
        value_text = row.get("OBS_VALUE")

        if value_text is None or value_text.strip() == "":
            raise ValueError("SDMX observation is missing OBS_VALUE.")

        structure_id = row.get("STRUCTURE_ID")

        if structure_id is None:
            raise ValueError("SDMX observation is missing STRUCTURE_ID.")

        time_period = row.get("TIME_PERIOD")

        if time_period is None or time_period.strip() == "":
            raise ValueError("SDMX observation is missing TIME_PERIOD.")

        dimensions = {
            name: value
            for name, value in row.items()
            if (name not in _NON_DIMENSION_COLUMNS and value is not None)
        }

        observations.append(
            Observation(
                dataset_code=_dataset_code_from_structure_id(structure_id),
                dimensions=dimensions,
                time_period=time_period,
                value=float(value_text),
            )
        )

    return tuple(observations)
