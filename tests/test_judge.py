import json
from datetime import UTC, datetime

import pytest

from eurostat_agent.judge import (
    ExpectedJudgeScores,
    JsonQualitativeJudge,
    JudgeAssessment,
    JudgeCalibrationCase,
    JudgeItem,
    JudgeReference,
    build_judge_prompt,
    build_judge_reference,
    calibrate_judge,
    judge_is_calibrated,
    judge_scores_pass,
    run_judge,
)
from eurostat_agent.judge_benchmark import (
    JUDGE_CALIBRATION_CASES,
)
from eurostat_agent.provenance import (
    ComputationProvenance,
    DeterministicAnswer,
    RetrievalProvenance,
)


def _reference() -> JudgeReference:
    return JudgeReference(
        value=123.0,
        unit="NR",
        dataset_code="DEMO_PJAN",
        filters=(
            ("TIME_PERIOD", "2024"),
            ("geo", "BE"),
        ),
        operation="none",
        source="Eurostat SDMX 3.0",
        source_url="https://example.test",
    )


def _item(
    response: str = "The value is 123 NR.",
) -> JudgeItem:
    return JudgeItem(
        question="What was the population in Belgium in 2024?",
        response=response,
        reference=_reference(),
    )


class FakeModel:
    def __init__(
        self,
        response: str,
    ) -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete(
        self,
        prompt: str,
    ) -> str:
        self.prompts.append(prompt)
        return self.response


def test_build_judge_reference_preserves_deterministic_truth() -> None:
    answer = DeterministicAnswer(
        value=123.0,
        unit="NR",
        computation=ComputationProvenance(
            operation="none",
            input_values=(123.0,),
            input_time_periods=("2024",),
            input_dimensions=(),
            output_value=123.0,
        ),
        provenance=RetrievalProvenance(
            dataset_code="DEMO_PJAN",
            dataset_agency="ESTAT",
            dataset_version="72.0",
            filters=(
                ("TIME_PERIOD", "2024"),
                ("geo", "BE"),
            ),
            retrieved_at=datetime(
                2026,
                9,
                16,
                12,
                0,
                tzinfo=UTC,
            ),
            data_updated_at="2026-09-15",
            source="Eurostat SDMX 3.0",
            source_url="https://example.test",
        ),
    )

    reference = build_judge_reference(answer)

    assert reference == JudgeReference(
        value=123.0,
        unit="NR",
        dataset_code="DEMO_PJAN",
        filters=(
            ("TIME_PERIOD", "2024"),
            ("geo", "BE"),
        ),
        operation="none",
        source="Eurostat SDMX 3.0",
        source_url="https://example.test",
    )


def test_build_judge_prompt_contains_deterministic_reference() -> None:
    prompt = build_judge_prompt(_item())

    assert "only source of factual truth" in prompt
    assert "DEMO_PJAN" in prompt
    assert "123.0" in prompt
    assert "TIME_PERIOD" in prompt
    assert "outside knowledge" in prompt


def test_json_judge_parses_valid_scores() -> None:
    model = FakeModel(
        json.dumps(
            {
                "relevance": 2,
                "factual_fidelity": 2,
                "provenance_fidelity": 2,
                "clarity": 2,
                "rationale": "Fully grounded.",
            }
        )
    )

    judge = JsonQualitativeJudge(model)

    assessment = judge.assess(_item())

    assert assessment == JudgeAssessment(
        relevance=2,
        factual_fidelity=2,
        provenance_fidelity=2,
        clarity=2,
        overall_pass=True,
        rationale="Fully grounded.",
    )

    assert len(model.prompts) == 1


def test_json_judge_derives_failure_deterministically() -> None:
    model = FakeModel(
        json.dumps(
            {
                "relevance": 2,
                "factual_fidelity": 2,
                "provenance_fidelity": 1,
                "clarity": 2,
                "rationale": "Provenance is incomplete.",
            }
        )
    )

    judge = JsonQualitativeJudge(model)

    assessment = judge.assess(_item())

    assert assessment.overall_pass is False


def test_json_judge_rejects_invalid_json() -> None:
    judge = JsonQualitativeJudge(FakeModel("not-json"))

    with pytest.raises(
        ValueError,
        match="Judge returned invalid JSON",
    ):
        judge.assess(_item())


def test_json_judge_rejects_unexpected_fields() -> None:
    model = FakeModel(
        json.dumps(
            {
                "relevance": 2,
                "factual_fidelity": 2,
                "provenance_fidelity": 2,
                "clarity": 2,
                "rationale": "Fine.",
                "overall_pass": True,
            }
        )
    )

    judge = JsonQualitativeJudge(model)

    with pytest.raises(
        ValueError,
        match="unexpected fields",
    ):
        judge.assess(_item())


def test_json_judge_rejects_invalid_score_range() -> None:
    model = FakeModel(
        json.dumps(
            {
                "relevance": 3,
                "factual_fidelity": 2,
                "provenance_fidelity": 2,
                "clarity": 2,
                "rationale": "Fine.",
            }
        )
    )

    judge = JsonQualitativeJudge(model)

    with pytest.raises(
        ValueError,
        match="invalid score values",
    ):
        judge.assess(_item())


def test_judge_scores_pass_requires_full_rubric() -> None:
    assert judge_scores_pass(
        relevance=2,
        factual_fidelity=2,
        provenance_fidelity=2,
        clarity=2,
    )

    assert not judge_scores_pass(
        relevance=2,
        factual_fidelity=2,
        provenance_fidelity=1,
        clarity=2,
    )


class SequenceJudge:
    def __init__(
        self,
        assessments: tuple[JudgeAssessment, ...],
    ) -> None:
        self._assessments = list(assessments)

    def assess(
        self,
        item: JudgeItem,
    ) -> JudgeAssessment:
        del item
        return self._assessments.pop(0)


def test_calibrate_judge_reports_agreement_metrics() -> None:
    cases = (
        JudgeCalibrationCase(
            item=_item("Complete answer"),
            expected=ExpectedJudgeScores(
                relevance=2,
                factual_fidelity=2,
                provenance_fidelity=2,
                clarity=2,
            ),
        ),
        JudgeCalibrationCase(
            item=_item("Partial answer"),
            expected=ExpectedJudgeScores(
                relevance=2,
                factual_fidelity=2,
                provenance_fidelity=0,
                clarity=2,
            ),
        ),
    )

    judge = SequenceJudge(
        (
            JudgeAssessment(
                relevance=2,
                factual_fidelity=2,
                provenance_fidelity=2,
                clarity=2,
                overall_pass=True,
                rationale="Exact.",
            ),
            JudgeAssessment(
                relevance=2,
                factual_fidelity=1,
                provenance_fidelity=0,
                clarity=2,
                overall_pass=False,
                rationale="One score differs.",
            ),
        )
    )

    summary = calibrate_judge(
        judge,
        cases,
    )

    assert summary.total_cases == 2
    assert summary.total_score_fields == 8
    assert summary.matching_score_fields == 7
    assert summary.score_accuracy == 0.875
    assert summary.pass_accuracy == 1.0
    assert summary.mean_absolute_error == 0.125


def test_calibrate_judge_rejects_empty_cases() -> None:
    judge = SequenceJudge(())

    with pytest.raises(
        ValueError,
        match="without calibration cases",
    ):
        calibrate_judge(
            judge,
            (),
        )


def test_judge_is_calibrated_uses_thresholds() -> None:
    judge = SequenceJudge(
        tuple(
            JudgeAssessment(
                relevance=case.expected.relevance,
                factual_fidelity=(case.expected.factual_fidelity),
                provenance_fidelity=(case.expected.provenance_fidelity),
                clarity=case.expected.clarity,
                overall_pass=case.expected.overall_pass,
                rationale="Matches anchor.",
            )
            for case in JUDGE_CALIBRATION_CASES
        )
    )

    summary = calibrate_judge(
        judge,
        JUDGE_CALIBRATION_CASES,
    )

    assert judge_is_calibrated(summary)


def test_run_judge_reports_results_and_failures() -> None:
    good_item = _item("Good")
    bad_item = _item("Bad")

    class SometimesJudge:
        def assess(
            self,
            item: JudgeItem,
        ) -> JudgeAssessment:
            if item.response == "Bad":
                raise ValueError("judge failed")

            return JudgeAssessment(
                relevance=2,
                factual_fidelity=2,
                provenance_fidelity=2,
                clarity=2,
                overall_pass=True,
                rationale="Good.",
            )

    run = run_judge(
        SometimesJudge(),
        (
            good_item,
            bad_item,
        ),
    )

    assert run.summary.total_cases == 2
    assert run.summary.completed_cases == 1
    assert run.summary.failed_cases == 1
    assert run.summary.completion_rate == 0.5
    assert run.summary.pass_rate == 1.0
    assert run.summary.mean_relevance == 2.0
    assert run.summary.mean_factual_fidelity == 2.0
    assert run.summary.mean_provenance_fidelity == 2.0
    assert run.summary.mean_clarity == 2.0

    assert len(run.results) == 1
    assert len(run.failures) == 1
    assert run.failures[0].error_type == "ValueError"
    assert run.failures[0].error_message == "judge failed"


def test_calibration_benchmark_contains_four_anchors() -> None:
    assert len(JUDGE_CALIBRATION_CASES) == 4

    assert JUDGE_CALIBRATION_CASES[0].expected.overall_pass is True

    assert JUDGE_CALIBRATION_CASES[1].expected.overall_pass is False

    assert JUDGE_CALIBRATION_CASES[2].expected.overall_pass is False

    assert JUDGE_CALIBRATION_CASES[3].expected.overall_pass is False
