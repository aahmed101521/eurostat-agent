from pathlib import Path

import pytest

from eurostat_agent.data import parse_sdmx_csv

FIXTURES = Path(__file__).parent / "fixtures" / "sdmx"


def test_parse_single_observation() -> None:
    csv_text = (FIXTURES / "demo_pjan_observation.csv").read_text(encoding="utf-8-sig")

    observations = parse_sdmx_csv(csv_text)

    assert len(observations) == 1

    observation = observations[0]

    assert observation.dataset_code == "DEMO_PJAN"
    assert observation.dimensions == {
        "freq": "A",
        "unit": "NR",
        "age": "Y20",
        "sex": "F",
        "geo": "BE",
    }
    assert observation.time_period == "2024"
    assert observation.value == 65231.0


def test_parse_empty_data_response() -> None:
    csv_text = "STRUCTURE,STRUCTURE_ID,freq,unit,age,sex,geo,TIME_PERIOD,OBS_VALUE\n"

    observations = parse_sdmx_csv(csv_text)

    assert observations == ()


def test_parse_rejects_missing_observation_value() -> None:
    csv_text = (
        "STRUCTURE,STRUCTURE_ID,freq,unit,age,sex,geo,"
        "TIME_PERIOD,OBS_VALUE\n"
        "dataflow,ESTAT:DEMO_PJAN(1.0),A,NR,Y20,F,BE,2024,\n"
    )

    with pytest.raises(
        ValueError,
        match="missing OBS_VALUE",
    ):
        parse_sdmx_csv(csv_text)
