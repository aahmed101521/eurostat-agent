from eurostat_agent.judge import (
    ExpectedJudgeScores,
    JudgeCalibrationCase,
    JudgeItem,
    JudgeReference,
)

_POPULATION_REFERENCE = JudgeReference(
    value=123.0,
    unit="NR",
    dataset_code="DEMO_PJAN",
    filters=(
        ("TIME_PERIOD", "2024"),
        ("age", "Y20"),
        ("freq", "A"),
        ("geo", "BE"),
        ("sex", "F"),
        ("unit", "NR"),
    ),
    operation="none",
    source="Eurostat SDMX 3.0",
    source_url=(
        "https://ec.europa.eu/eurostat/api/"
        "dissemination/sdmx/3.0/data/dataflow/"
        "ESTAT/DEMO_PJAN/72.0"
    ),
)


JUDGE_CALIBRATION_CASES = (
    JudgeCalibrationCase(
        item=JudgeItem(
            question=(
                "What was the population of 20-year-old women in Belgium in 2024?"
            ),
            response=(
                "The value is 123 NR. Source: Eurostat SDMX 3.0, "
                "dataset DEMO_PJAN, using TIME_PERIOD=2024, "
                "age=Y20, freq=A, geo=BE, sex=F, and unit=NR."
            ),
            reference=_POPULATION_REFERENCE,
        ),
        expected=ExpectedJudgeScores(
            relevance=2,
            factual_fidelity=2,
            provenance_fidelity=2,
            clarity=2,
        ),
    ),
    JudgeCalibrationCase(
        item=JudgeItem(
            question=(
                "What was the population of 20-year-old women in Belgium in 2024?"
            ),
            response=("The population was 123 NR."),
            reference=_POPULATION_REFERENCE,
        ),
        expected=ExpectedJudgeScores(
            relevance=2,
            factual_fidelity=2,
            provenance_fidelity=0,
            clarity=2,
        ),
    ),
    JudgeCalibrationCase(
        item=JudgeItem(
            question=(
                "What was the population of 20-year-old women in Belgium in 2024?"
            ),
            response=(
                "The value is 999 NR. Source: Eurostat SDMX 3.0, "
                "dataset DEMO_PJAN, using TIME_PERIOD=2024, "
                "age=Y20, freq=A, geo=BE, sex=F, and unit=NR."
            ),
            reference=_POPULATION_REFERENCE,
        ),
        expected=ExpectedJudgeScores(
            relevance=2,
            factual_fidelity=0,
            provenance_fidelity=2,
            clarity=2,
        ),
    ),
    JudgeCalibrationCase(
        item=JudgeItem(
            question=(
                "What was the population of 20-year-old women in Belgium in 2024?"
            ),
            response=("I cannot answer that question."),
            reference=_POPULATION_REFERENCE,
        ),
        expected=ExpectedJudgeScores(
            relevance=0,
            factual_fidelity=0,
            provenance_fidelity=0,
            clarity=2,
        ),
    ),
)
