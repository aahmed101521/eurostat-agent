"""Stage 11 end-to-end robustness evaluation and deterministic reporting."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Literal

from eurostat_agent.cache import MetadataCachingClient
from eurostat_agent.catalogue import DatasetIndex, DatasetRecord
from eurostat_agent.controller import (
    ControllerClient,
    DatasetSelector,
    QuestionPlan,
    QuestionPlanner,
    answer_planned_question,
)
from eurostat_agent.evaluation import (
    BenchmarkCase,
    BenchmarkScore,
    score_deterministic_answer,
)
from eurostat_agent.experiments import ExperimentConfig
from eurostat_agent.provenance import DeterministicAnswer
from eurostat_agent.reporting import (
    OperationalReport,
    build_operational_report,
    operational_report_to_dict,
)
from eurostat_agent.tracing import (
    ExecutionTrace,
    TraceRecorder,
    execution_trace_to_dict,
)

CaseCategory = Literal[
    "direct_supported",
    "linguistic_variation",
    "deterministic_computation",
    "temporal_semantics",
    "multi_value_unsupported",
    "safe_failure",
]
ExpectedCapability = Literal["supported", "known_unsupported", "safe_failure"]
OutcomeClassification = Literal[
    "completed_exact",
    "completed_inexact",
    "model_completion_failure",
    "deterministic_validation_failure",
    "expected_architecture_limitation",
    "safe_failure",
    "unexpected_failure",
]

_ANNUAL_PERIOD = re.compile(r"^\d{4}$")
_FULL_DATE_PERIOD = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MODEL_FAILURE_STAGES = {
    "planner_model_call",
    "selector_model_call",
    "planning",
    "dataset_selection",
}
_DETERMINISTIC_FAILURE_STAGES = {
    "catalogue_search",
    "filter_resolution",
    "retrieval",
    "computation",
}


@dataclass(frozen=True)
class RobustnessCase:
    """One Stage 11 case with explicit capability and expected semantics."""

    case_id: str
    category: CaseCategory
    question: str
    expected_capability: ExpectedCapability
    expected_dataset_code: str | None = None
    expected_filters: tuple[tuple[str, str], ...] = ()
    expected_operation: str | None = None
    score_filters: bool = True

    def benchmark_case(self) -> BenchmarkCase:
        """Convert a scoreable case to the existing Stage 7 benchmark type."""

        if self.expected_dataset_code is None or self.expected_operation is None:
            raise ValueError(f"Case {self.case_id!r} is not benchmark-scoreable.")

        return BenchmarkCase(
            question=self.question,
            expected_dataset_code=self.expected_dataset_code,
            expected_filters=self.expected_filters,
            expected_operation=self.expected_operation,
            score_filters=self.score_filters,
        )


@dataclass(frozen=True)
class StageLatencySummary:
    """Observed latency distribution for one trace stage."""

    stage: str
    observation_count: int
    median_seconds: float
    p95_seconds: float | None


@dataclass(frozen=True)
class TemporalBehaviorSummary:
    """Observed planner treatment of Stage 11 temporal-semantics cases."""

    total_cases: int
    planned_cases: int
    annual_year_values: int
    full_date_values: int
    other_values: int
    missing_values: int


@dataclass(frozen=True)
class RobustnessEvidence:
    """Complete case-level Stage 11 evidence for one model execution."""

    case: RobustnessCase
    question_plan: QuestionPlan | None
    selected_dataset_code: str | None
    answer: DeterministicAnswer | None
    benchmark_score: BenchmarkScore | None
    trace: ExecutionTrace
    operational_report: OperationalReport
    outcome: OutcomeClassification
    failure_stage: str | None
    error_type: str | None
    error_message: str | None
    provenance_violation: bool


@dataclass(frozen=True)
class RobustnessSummary:
    """Aggregate Stage 11 statistics derived only from case-level evidence."""

    total_cases: int
    completed_cases: int
    completion_rate: float
    supported_cases: int
    completed_supported_cases: int
    supported_case_completion_rate: float
    exact_supported_cases: int
    supported_case_exact_match: float
    known_unsupported_cases: int
    safe_failure_cases: int
    safe_failures: int
    safe_failure_rate: float
    architecture_limitation_count: int
    completed_inexact_count: int
    model_completion_failure_count: int
    deterministic_validation_failure_count: int
    unexpected_failure_count: int
    provenance_violation_count: int
    dataset_accuracy: float
    filters_accuracy: float
    operation_accuracy: float
    exact_match_accuracy: float
    outcome_distribution: tuple[tuple[str, int], ...]
    failure_stage_distribution: tuple[tuple[str, int], ...]
    median_total_latency_seconds: float | None
    p95_total_latency_seconds: float | None
    stage_latencies: tuple[StageLatencySummary, ...]
    model_call_count: int
    cache_hits: int
    cache_misses: int
    temporal_behavior: TemporalBehaviorSummary


@dataclass(frozen=True)
class RobustnessRun:
    """Stage 11 run for one model configuration."""

    config: ExperimentConfig
    evidence: tuple[RobustnessEvidence, ...]
    summary: RobustnessSummary


@dataclass(frozen=True)
class RobustnessConfigFailure:
    """Failure that prevented a model configuration from producing a run."""

    config: ExperimentConfig
    error_type: str
    error_message: str


@dataclass(frozen=True)
class RobustnessComparison:
    """Stage 11 runs and any configuration-level failures."""

    runs: tuple[RobustnessRun, ...]
    failures: tuple[RobustnessConfigFailure, ...]


class _RecordingPlanner:
    def __init__(
        self,
        delegate: QuestionPlanner,
        *,
        recorder: TraceRecorder,
        model_metadata: tuple[tuple[str, str], ...],
    ) -> None:
        self._delegate = delegate
        self._recorder = recorder
        self._model_metadata = model_metadata
        self.plan_result: QuestionPlan | None = None

    def plan(self, question: str) -> QuestionPlan:
        with self._recorder.stage(
            "planner_model_call",
            metadata=self._model_metadata,
        ):
            plan = self._delegate.plan(question)

        self.plan_result = plan
        return plan


class _RecordingSelector:
    def __init__(
        self,
        delegate: DatasetSelector,
        *,
        recorder: TraceRecorder,
        model_metadata: tuple[tuple[str, str], ...],
    ) -> None:
        self._delegate = delegate
        self._recorder = recorder
        self._model_metadata = model_metadata
        self.selected_code: str | None = None

    def select_dataset(
        self,
        question: str,
        candidates: tuple[DatasetRecord, ...],
    ) -> str:
        with self._recorder.stage(
            "selector_model_call",
            metadata=self._model_metadata,
        ):
            selected_code = self._delegate.select_dataset(question, candidates)

        self.selected_code = selected_code
        return selected_code


def _model_metadata(config: ExperimentConfig) -> tuple[tuple[str, str], ...]:
    return (
        ("model", config.model),
        ("backend", "ollama"),
        ("prompt_variant", config.prompt_variant),
    )


def _specific_failure_stage(report: OperationalReport) -> str | None:
    for failure in report.failures:
        if failure.stage != "controller_total":
            return failure.stage

    if report.failures:
        return report.failures[0].stage

    return None


def _external_request_failure(
    error_type: str | None,
    error_message: str | None,
) -> bool:
    if error_type not in {"EurostatClientError", "ModelClientError"}:
        return False

    return error_message is not None and "request failed" in error_message.casefold()


def _supported_failure_outcome(
    *,
    failure_stage: str | None,
    error_type: str | None,
    error_message: str | None,
) -> OutcomeClassification:
    if _external_request_failure(error_type, error_message):
        return "unexpected_failure"

    if failure_stage in _MODEL_FAILURE_STAGES:
        if error_message == "No dataset candidates were found.":
            return "deterministic_validation_failure"

        return "model_completion_failure"

    if failure_stage in _DETERMINISTIC_FAILURE_STAGES:
        return "deterministic_validation_failure"

    return "unexpected_failure"


def has_complete_provenance(answer: DeterministicAnswer) -> bool:
    """Validate the minimum statistical and computation provenance contract."""

    provenance = answer.provenance
    computation = answer.computation

    required_text = (
        answer.unit,
        provenance.dataset_code,
        provenance.dataset_agency,
        provenance.dataset_version,
        provenance.data_updated_at,
        provenance.source,
        provenance.source_url,
        computation.operation,
    )

    retrieved_at = provenance.retrieved_at
    timestamp_is_aware = (
        retrieved_at.tzinfo is not None and retrieved_at.utcoffset() is not None
    )
    computation_shapes_match = (
        bool(computation.input_values)
        and len(computation.input_values) == len(computation.input_time_periods)
        and len(computation.input_values) == len(computation.input_dimensions)
    )

    return (
        all(bool(value) for value in required_text)
        and timestamp_is_aware
        and computation_shapes_match
        and answer.value == computation.output_value
    )


def _classify_outcome(
    case: RobustnessCase,
    *,
    question_plan: QuestionPlan | None,
    answer: DeterministicAnswer | None,
    score: BenchmarkScore | None,
    provenance_violation: bool,
    failure_stage: str | None,
    error_type: str | None,
    error_message: str | None,
) -> OutcomeClassification:
    if provenance_violation:
        return "unexpected_failure"

    if case.expected_capability == "safe_failure":
        if answer is None and not _external_request_failure(error_type, error_message):
            return "safe_failure"

        return "unexpected_failure"

    if case.expected_capability == "known_unsupported":
        if answer is not None:
            return "completed_inexact"

        if _external_request_failure(error_type, error_message):
            return "unexpected_failure"

        if question_plan is not None:
            return "expected_architecture_limitation"

        return _supported_failure_outcome(
            failure_stage=failure_stage,
            error_type=error_type,
            error_message=error_message,
        )

    if answer is not None:
        if score is not None and score.exact_match:
            return "completed_exact"

        return "completed_inexact"

    return _supported_failure_outcome(
        failure_stage=failure_stage,
        error_type=error_type,
        error_message=error_message,
    )


def run_robustness_case(
    case: RobustnessCase,
    *,
    config: ExperimentConfig,
    planner: QuestionPlanner,
    selector: DatasetSelector,
    client: ControllerClient,
    index: DatasetIndex,
    retrieved_at: datetime,
    limit: int = 10,
) -> RobustnessEvidence:
    """Run one Stage 11 case through the unchanged deterministic controller."""

    recorder = TraceRecorder(execution_id=f"{config.name}:{case.case_id}")
    metadata = _model_metadata(config)
    recording_planner = _RecordingPlanner(
        planner,
        recorder=recorder,
        model_metadata=metadata,
    )
    recording_selector = _RecordingSelector(
        selector,
        recorder=recorder,
        model_metadata=metadata,
    )
    caching_client = MetadataCachingClient(
        client,
        trace_recorder=recorder,
    )

    answer: DeterministicAnswer | None = None
    error_type: str | None = None
    error_message: str | None = None

    try:
        answer = answer_planned_question(
            recording_planner,
            recording_selector,
            caching_client,
            case.question,
            index=index,
            retrieved_at=retrieved_at,
            limit=limit,
            trace_recorder=recorder,
        )
    except Exception as exc:
        error_type = type(exc).__name__
        error_message = str(exc)

    trace = recorder.trace
    operational_report = build_operational_report(trace)
    failure_stage = _specific_failure_stage(operational_report)
    provenance_violation = answer is not None and not has_complete_provenance(answer)

    score: BenchmarkScore | None = None
    if answer is not None and case.expected_capability == "supported":
        score = score_deterministic_answer(case.benchmark_case(), answer)

    outcome = _classify_outcome(
        case,
        question_plan=recording_planner.plan_result,
        answer=answer,
        score=score,
        provenance_violation=provenance_violation,
        failure_stage=failure_stage,
        error_type=error_type,
        error_message=error_message,
    )

    return RobustnessEvidence(
        case=case,
        question_plan=recording_planner.plan_result,
        selected_dataset_code=recording_selector.selected_code,
        answer=answer,
        benchmark_score=score,
        trace=trace,
        operational_report=operational_report,
        outcome=outcome,
        failure_stage=failure_stage,
        error_type=error_type,
        error_message=error_message,
        provenance_violation=provenance_violation,
    )


def _p95(values: tuple[float, ...]) -> float | None:
    """Return nearest-rank empirical p95 once at least 20 observations exist."""

    if len(values) < 20:
        return None

    ordered = sorted(values)
    rank = math.ceil(0.95 * len(ordered))
    return ordered[rank - 1]


def _accuracy(
    scores: tuple[BenchmarkScore, ...],
    attribute: Literal[
        "dataset_match",
        "filters_match",
        "operation_match",
        "exact_match",
    ],
) -> float:
    if not scores:
        return 0.0

    if attribute == "filters_match":
        filtered_scores = tuple(score for score in scores if score.filters_scored)
        if not filtered_scores:
            return 0.0

        matches = tuple(score.filters_match for score in filtered_scores)
        return sum(matches) / len(matches)

    if attribute == "dataset_match":
        matches = tuple(score.dataset_match for score in scores)
    elif attribute == "operation_match":
        matches = tuple(score.operation_match for score in scores)
    else:
        matches = tuple(score.exact_match for score in scores)

    return sum(matches) / len(matches)


def _temporal_behavior(
    evidence: tuple[RobustnessEvidence, ...],
) -> TemporalBehaviorSummary:
    temporal = tuple(
        item for item in evidence if item.case.category == "temporal_semantics"
    )
    planned = 0
    annual = 0
    full_date = 0
    other = 0
    missing = 0

    for item in temporal:
        if item.question_plan is None:
            missing += 1
            continue

        planned += 1
        value = item.question_plan.filters.get("TIME_PERIOD")
        if value is None:
            missing += 1
        elif _ANNUAL_PERIOD.fullmatch(value):
            annual += 1
        elif _FULL_DATE_PERIOD.fullmatch(value):
            full_date += 1
        else:
            other += 1

    return TemporalBehaviorSummary(
        total_cases=len(temporal),
        planned_cases=planned,
        annual_year_values=annual,
        full_date_values=full_date,
        other_values=other,
        missing_values=missing,
    )


def summarize_robustness_evidence(
    evidence: tuple[RobustnessEvidence, ...],
) -> RobustnessSummary:
    """Build the aggregate Stage 11 report from preserved case-level evidence."""

    total_cases = len(evidence)
    completed = tuple(
        item
        for item in evidence
        if item.answer is not None and not item.provenance_violation
    )
    supported = tuple(
        item for item in evidence if item.case.expected_capability == "supported"
    )
    completed_supported = tuple(
        item
        for item in supported
        if item.answer is not None and not item.provenance_violation
    )
    exact_supported = tuple(
        item
        for item in completed_supported
        if item.benchmark_score is not None and item.benchmark_score.exact_match
    )
    known_unsupported = tuple(
        item
        for item in evidence
        if item.case.expected_capability == "known_unsupported"
    )
    safe_failure_cases = tuple(
        item for item in evidence if item.case.expected_capability == "safe_failure"
    )
    safe_failures = tuple(
        item for item in safe_failure_cases if item.outcome == "safe_failure"
    )
    scores = tuple(
        item.benchmark_score
        for item in completed_supported
        if item.benchmark_score is not None
    )

    total_durations = tuple(
        item.operational_report.total_duration_seconds
        for item in evidence
        if item.operational_report.total_duration_seconds is not None
    )
    stage_values: dict[str, list[float]] = defaultdict(list)
    for item in evidence:
        execution_stage_totals: dict[str, float] = defaultdict(float)
        for stage, duration in item.operational_report.stage_durations:
            execution_stage_totals[stage] += duration

        for stage, duration in execution_stage_totals.items():
            stage_values[stage].append(duration)

    stage_latencies = tuple(
        StageLatencySummary(
            stage=stage,
            observation_count=len(values),
            median_seconds=median(values),
            p95_seconds=_p95(tuple(values)),
        )
        for stage, values in sorted(stage_values.items())
        if values
    )

    outcomes = Counter(item.outcome for item in evidence)
    failure_stages = Counter(
        item.failure_stage for item in evidence if item.failure_stage is not None
    )

    return RobustnessSummary(
        total_cases=total_cases,
        completed_cases=len(completed),
        completion_rate=(len(completed) / total_cases if total_cases else 0.0),
        supported_cases=len(supported),
        completed_supported_cases=len(completed_supported),
        supported_case_completion_rate=(
            len(completed_supported) / len(supported) if supported else 0.0
        ),
        exact_supported_cases=len(exact_supported),
        supported_case_exact_match=(
            len(exact_supported) / len(supported) if supported else 0.0
        ),
        known_unsupported_cases=len(known_unsupported),
        safe_failure_cases=len(safe_failure_cases),
        safe_failures=len(safe_failures),
        safe_failure_rate=(
            len(safe_failures) / len(safe_failure_cases) if safe_failure_cases else 0.0
        ),
        architecture_limitation_count=outcomes["expected_architecture_limitation"],
        completed_inexact_count=outcomes["completed_inexact"],
        model_completion_failure_count=outcomes["model_completion_failure"],
        deterministic_validation_failure_count=outcomes[
            "deterministic_validation_failure"
        ],
        unexpected_failure_count=outcomes["unexpected_failure"],
        provenance_violation_count=sum(item.provenance_violation for item in evidence),
        dataset_accuracy=_accuracy(scores, "dataset_match"),
        filters_accuracy=_accuracy(scores, "filters_match"),
        operation_accuracy=_accuracy(scores, "operation_match"),
        exact_match_accuracy=_accuracy(scores, "exact_match"),
        outcome_distribution=tuple(sorted(outcomes.items())),
        failure_stage_distribution=tuple(sorted(failure_stages.items())),
        median_total_latency_seconds=(
            median(total_durations) if total_durations else None
        ),
        p95_total_latency_seconds=_p95(total_durations),
        stage_latencies=stage_latencies,
        model_call_count=sum(
            item.operational_report.model_call_count for item in evidence
        ),
        cache_hits=sum(item.operational_report.cache_hits for item in evidence),
        cache_misses=sum(item.operational_report.cache_misses for item in evidence),
        temporal_behavior=_temporal_behavior(evidence),
    )


def run_robustness_benchmark(
    cases: tuple[RobustnessCase, ...],
    *,
    config: ExperimentConfig,
    planner: QuestionPlanner,
    selector: DatasetSelector,
    client: ControllerClient,
    index: DatasetIndex,
    retrieved_at: datetime,
    limit: int = 10,
) -> RobustnessRun:
    """Run all Stage 11 cases for one model configuration."""

    evidence = tuple(
        run_robustness_case(
            case,
            config=config,
            planner=planner,
            selector=selector,
            client=client,
            index=index,
            retrieved_at=retrieved_at,
            limit=limit,
        )
        for case in cases
    )

    return RobustnessRun(
        config=config,
        evidence=evidence,
        summary=summarize_robustness_evidence(evidence),
    )


def robustness_case_to_dict(case: RobustnessCase) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "category": case.category,
        "question": case.question,
        "expected_capability": case.expected_capability,
        "expected_dataset_code": case.expected_dataset_code,
        "expected_filters": [list(item) for item in case.expected_filters],
        "expected_operation": case.expected_operation,
        "score_filters": case.score_filters,
    }


def question_plan_to_dict(plan: QuestionPlan | None) -> dict[str, object] | None:
    if plan is None:
        return None

    return {
        "dataset_query": plan.dataset_query,
        "filters": dict(sorted(plan.filters.items())),
        "operation": plan.operation,
    }


def benchmark_score_to_dict(score: BenchmarkScore | None) -> dict[str, object] | None:
    if score is None:
        return None

    return {
        "dataset_match": score.dataset_match,
        "filters_match": score.filters_match,
        "operation_match": score.operation_match,
        "exact_match": score.exact_match,
        "filters_scored": score.filters_scored,
    }


def deterministic_answer_to_dict(
    answer: DeterministicAnswer | None,
) -> dict[str, object] | None:
    if answer is None:
        return None

    provenance = answer.provenance
    computation = answer.computation
    return {
        "value": answer.value,
        "unit": answer.unit,
        "provenance": {
            "dataset_code": provenance.dataset_code,
            "dataset_agency": provenance.dataset_agency,
            "dataset_version": provenance.dataset_version,
            "filters": [list(item) for item in provenance.filters],
            "retrieved_at": provenance.retrieved_at.isoformat(),
            "data_updated_at": provenance.data_updated_at,
            "source": provenance.source,
            "source_url": provenance.source_url,
        },
        "computation": {
            "operation": computation.operation,
            "input_values": list(computation.input_values),
            "input_time_periods": list(computation.input_time_periods),
            "input_dimensions": [
                [list(item) for item in dimensions]
                for dimensions in computation.input_dimensions
            ],
            "output_value": computation.output_value,
        },
    }


def robustness_evidence_to_dict(
    evidence: RobustnessEvidence,
) -> dict[str, object]:
    return {
        "case": robustness_case_to_dict(evidence.case),
        "question_plan": question_plan_to_dict(evidence.question_plan),
        "selected_dataset_code": evidence.selected_dataset_code,
        "answer": deterministic_answer_to_dict(evidence.answer),
        "benchmark_score": benchmark_score_to_dict(evidence.benchmark_score),
        "outcome": evidence.outcome,
        "failure_stage": evidence.failure_stage,
        "error_type": evidence.error_type,
        "error_message": evidence.error_message,
        "provenance_violation": evidence.provenance_violation,
        "execution_trace": execution_trace_to_dict(evidence.trace),
        "operational_report": operational_report_to_dict(evidence.operational_report),
    }


def _stage_latency_to_dict(summary: StageLatencySummary) -> dict[str, object]:
    return {
        "stage": summary.stage,
        "observation_count": summary.observation_count,
        "median_seconds": summary.median_seconds,
        "p95_seconds": summary.p95_seconds,
    }


def _temporal_behavior_to_dict(
    summary: TemporalBehaviorSummary,
) -> dict[str, int]:
    return {
        "total_cases": summary.total_cases,
        "planned_cases": summary.planned_cases,
        "annual_year_values": summary.annual_year_values,
        "full_date_values": summary.full_date_values,
        "other_values": summary.other_values,
        "missing_values": summary.missing_values,
    }


def robustness_summary_to_dict(summary: RobustnessSummary) -> dict[str, object]:
    return {
        "total_cases": summary.total_cases,
        "completed_cases": summary.completed_cases,
        "completion_rate": summary.completion_rate,
        "supported_cases": summary.supported_cases,
        "completed_supported_cases": summary.completed_supported_cases,
        "supported_case_completion_rate": summary.supported_case_completion_rate,
        "exact_supported_cases": summary.exact_supported_cases,
        "supported_case_exact_match": summary.supported_case_exact_match,
        "known_unsupported_cases": summary.known_unsupported_cases,
        "safe_failure_cases": summary.safe_failure_cases,
        "safe_failures": summary.safe_failures,
        "safe_failure_rate": summary.safe_failure_rate,
        "architecture_limitation_count": summary.architecture_limitation_count,
        "completed_inexact_count": summary.completed_inexact_count,
        "model_completion_failure_count": summary.model_completion_failure_count,
        "deterministic_validation_failure_count": (
            summary.deterministic_validation_failure_count
        ),
        "unexpected_failure_count": summary.unexpected_failure_count,
        "provenance_violation_count": summary.provenance_violation_count,
        "dataset_accuracy": summary.dataset_accuracy,
        "filters_accuracy": summary.filters_accuracy,
        "operation_accuracy": summary.operation_accuracy,
        "exact_match_accuracy": summary.exact_match_accuracy,
        "outcome_distribution": [list(item) for item in summary.outcome_distribution],
        "failure_stage_distribution": [
            list(item) for item in summary.failure_stage_distribution
        ],
        "median_total_latency_seconds": summary.median_total_latency_seconds,
        "p95_total_latency_seconds": summary.p95_total_latency_seconds,
        "stage_latencies": [
            _stage_latency_to_dict(item) for item in summary.stage_latencies
        ],
        "model_call_count": summary.model_call_count,
        "cache_hits": summary.cache_hits,
        "cache_misses": summary.cache_misses,
        "temporal_behavior": _temporal_behavior_to_dict(summary.temporal_behavior),
    }


def robustness_run_to_dict(run: RobustnessRun) -> dict[str, object]:
    return {
        "config": {
            "name": run.config.name,
            "model": run.config.model,
            "prompt_variant": run.config.prompt_variant,
        },
        "summary": robustness_summary_to_dict(run.summary),
        "cases": [robustness_evidence_to_dict(item) for item in run.evidence],
    }


def robustness_comparison_to_dict(
    comparison: RobustnessComparison,
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "runs": [robustness_run_to_dict(run) for run in comparison.runs],
        "failures": [
            {
                "config": {
                    "name": failure.config.name,
                    "model": failure.config.model,
                    "prompt_variant": failure.config.prompt_variant,
                },
                "error_type": failure.error_type,
                "error_message": failure.error_message,
            }
            for failure in comparison.failures
        ],
    }


def write_robustness_report(
    comparison: RobustnessComparison,
    path: Path,
) -> None:
    """Write the complete Stage 11 report using deterministic JSON ordering."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            robustness_comparison_to_dict(comparison),
            indent=2,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
