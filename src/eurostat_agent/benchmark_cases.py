"""Stratified Stage 11 end-to-end robustness benchmark cases."""

from eurostat_agent.robustness import RobustnessCase


def _population_filters(
    *,
    year: str,
    age: str,
    sex: str,
    geo: str,
) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (
                ("TIME_PERIOD", year),
                ("age", age),
                ("freq", "A"),
                ("geo", geo),
                ("sex", sex),
                ("unit", "NR"),
            )
        )
    )


_BE_F_Y20_2024 = _population_filters(
    year="2024",
    age="Y20",
    sex="F",
    geo="BE",
)


ROBUSTNESS_BENCHMARK_CASES = (
    # Category A — direct supported lookups.
    RobustnessCase(
        case_id="A01",
        category="direct_supported",
        question="What was the population of 20-year-old women in Belgium in 2024?",
        expected_capability="supported",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=_BE_F_Y20_2024,
        expected_operation="none",
    ),
    RobustnessCase(
        case_id="A02",
        category="direct_supported",
        question="What was the population of 20-year-old men in Germany in 2024?",
        expected_capability="supported",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=_population_filters(
            year="2024",
            age="Y20",
            sex="M",
            geo="DE",
        ),
        expected_operation="none",
    ),
    RobustnessCase(
        case_id="A03",
        category="direct_supported",
        question="How many 30-year-old women lived in France in 2023?",
        expected_capability="supported",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=_population_filters(
            year="2023",
            age="Y30",
            sex="F",
            geo="FR",
        ),
        expected_operation="none",
    ),
    RobustnessCase(
        case_id="A04",
        category="direct_supported",
        question="How many 40-year-old men lived in Italy in 2022?",
        expected_capability="supported",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=_population_filters(
            year="2022",
            age="Y40",
            sex="M",
            geo="IT",
        ),
        expected_operation="none",
    ),
    RobustnessCase(
        case_id="A05",
        category="direct_supported",
        question="What was the population of 25-year-old women in Spain in 2024?",
        expected_capability="supported",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=_population_filters(
            year="2024",
            age="Y25",
            sex="F",
            geo="ES",
        ),
        expected_operation="none",
    ),
    RobustnessCase(
        case_id="A06",
        category="direct_supported",
        question=(
            "What was the population of 35-year-old men in the Netherlands in 2023?"
        ),
        expected_capability="supported",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=_population_filters(
            year="2023",
            age="Y35",
            sex="M",
            geo="NL",
        ),
        expected_operation="none",
    ),
    # Category B — linguistic variation over the same semantics.
    RobustnessCase(
        case_id="B01",
        category="linguistic_variation",
        question="How many females aged 20 lived in Belgium in 2024?",
        expected_capability="supported",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=_BE_F_Y20_2024,
        expected_operation="none",
    ),
    RobustnessCase(
        case_id="B02",
        category="linguistic_variation",
        question="What was Belgium's female population at age 20 in 2024?",
        expected_capability="supported",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=_BE_F_Y20_2024,
        expected_operation="none",
    ),
    RobustnessCase(
        case_id="B03",
        category="linguistic_variation",
        question="How many Belgian women were 20 years old in 2024?",
        expected_capability="supported",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=_BE_F_Y20_2024,
        expected_operation="none",
    ),
    RobustnessCase(
        case_id="B04",
        category="linguistic_variation",
        question="Give the 2024 population of 20-year-old females in Belgium.",
        expected_capability="supported",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=_BE_F_Y20_2024,
        expected_operation="none",
    ),
    RobustnessCase(
        case_id="B05",
        category="linguistic_variation",
        question="For Belgium in 2024, how many women were age 20?",
        expected_capability="supported",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=_BE_F_Y20_2024,
        expected_operation="none",
    ),
    RobustnessCase(
        case_id="B06",
        category="linguistic_variation",
        question="What is the female population aged 20 for Belgium, 2024?",
        expected_capability="supported",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=_BE_F_Y20_2024,
        expected_operation="none",
    ),
    # Category C — deterministic computation intent.
    # The two single-observation sums are end-to-end representable today.
    # Pairwise operations are retained as known limitations because the current
    # QuestionPlan cannot safely name both operands within one dimension.
    RobustnessCase(
        case_id="C01",
        category="deterministic_computation",
        question=(
            "What is the sum of the population of 20-year-old women in Belgium in 2024?"
        ),
        expected_capability="supported",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=_BE_F_Y20_2024,
        expected_operation="sum",
    ),
    RobustnessCase(
        case_id="C02",
        category="deterministic_computation",
        question=(
            "Add up the population observation for women aged 20 in Belgium for 2024."
        ),
        expected_capability="supported",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=_BE_F_Y20_2024,
        expected_operation="sum",
    ),
    RobustnessCase(
        case_id="C03",
        category="deterministic_computation",
        question=(
            "What is the difference between the population of 20-year-old women "
            "in France and Germany in 2024?"
        ),
        expected_capability="known_unsupported",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(
            ("TIME_PERIOD", "2024"),
            ("age", "Y20"),
            ("freq", "A"),
            ("geo", "FR"),
            ("geo", "DE"),
            ("sex", "F"),
            ("unit", "NR"),
        ),
        expected_operation="difference",
    ),
    RobustnessCase(
        case_id="C04",
        category="deterministic_computation",
        question=(
            "What is the ratio of the population of 20-year-old women in France "
            "to Germany in 2024?"
        ),
        expected_capability="known_unsupported",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(
            ("TIME_PERIOD", "2024"),
            ("age", "Y20"),
            ("freq", "A"),
            ("geo", "FR"),
            ("geo", "DE"),
            ("sex", "F"),
            ("unit", "NR"),
        ),
        expected_operation="ratio",
    ),
    RobustnessCase(
        case_id="C05",
        category="deterministic_computation",
        question=(
            "What was the percentage change in the population of 20-year-old "
            "women in Belgium from 2023 to 2024?"
        ),
        expected_capability="known_unsupported",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(
            ("TIME_PERIOD", "2023"),
            ("TIME_PERIOD", "2024"),
            ("age", "Y20"),
            ("freq", "A"),
            ("geo", "BE"),
            ("sex", "F"),
            ("unit", "NR"),
        ),
        expected_operation="percentage_change",
    ),
    RobustnessCase(
        case_id="C06",
        category="deterministic_computation",
        question=(
            "What is the difference between the female and male populations aged "
            "20 in Belgium in 2024?"
        ),
        expected_capability="known_unsupported",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(
            ("TIME_PERIOD", "2024"),
            ("age", "Y20"),
            ("freq", "A"),
            ("geo", "BE"),
            ("sex", "F"),
            ("sex", "M"),
            ("unit", "NR"),
        ),
        expected_operation="difference",
    ),
    # Category D — temporal semantics.
    RobustnessCase(
        case_id="D01",
        category="temporal_semantics",
        question="How many women aged 20 lived in Belgium in 2024?",
        expected_capability="supported",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=_BE_F_Y20_2024,
        expected_operation="none",
    ),
    RobustnessCase(
        case_id="D02",
        category="temporal_semantics",
        question="How many women aged 20 lived in Belgium for 2024?",
        expected_capability="supported",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=_BE_F_Y20_2024,
        expected_operation="none",
    ),
    RobustnessCase(
        case_id="D03",
        category="temporal_semantics",
        question="How many women aged 20 lived in Belgium on 1 January 2024?",
        expected_capability="supported",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=_BE_F_Y20_2024,
        expected_operation="none",
    ),
    RobustnessCase(
        case_id="D04",
        category="temporal_semantics",
        question=("How many women aged 20 lived in Belgium at the beginning of 2024?"),
        expected_capability="supported",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=_BE_F_Y20_2024,
        expected_operation="none",
    ),
    RobustnessCase(
        case_id="D05",
        category="temporal_semantics",
        question="What was Belgium's 2024 population of women aged 20?",
        expected_capability="supported",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=_BE_F_Y20_2024,
        expected_operation="none",
    ),
    RobustnessCase(
        case_id="D06",
        category="temporal_semantics",
        question="How many 20-year-old women lived in Belgium as of January 1, 2024?",
        expected_capability="supported",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=_BE_F_Y20_2024,
        expected_operation="none",
    ),
    # Category E — currently unsupported multi-value requests.
    RobustnessCase(
        case_id="E01",
        category="multi_value_unsupported",
        question=(
            "Give the population of 20-year-old women in Belgium for 2023 and 2024."
        ),
        expected_capability="known_unsupported",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(
            ("TIME_PERIOD", "2023"),
            ("TIME_PERIOD", "2024"),
            ("age", "Y20"),
            ("freq", "A"),
            ("geo", "BE"),
            ("sex", "F"),
            ("unit", "NR"),
        ),
        expected_operation="none",
    ),
    RobustnessCase(
        case_id="E02",
        category="multi_value_unsupported",
        question=(
            "Give the population of 20-year-old women in France and Germany in 2024."
        ),
        expected_capability="known_unsupported",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(
            ("TIME_PERIOD", "2024"),
            ("age", "Y20"),
            ("freq", "A"),
            ("geo", "FR"),
            ("geo", "DE"),
            ("sex", "F"),
            ("unit", "NR"),
        ),
        expected_operation="none",
    ),
    RobustnessCase(
        case_id="E03",
        category="multi_value_unsupported",
        question="Give the male and female populations aged 20 in Belgium in 2024.",
        expected_capability="known_unsupported",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(
            ("TIME_PERIOD", "2024"),
            ("age", "Y20"),
            ("freq", "A"),
            ("geo", "BE"),
            ("sex", "M"),
            ("sex", "F"),
            ("unit", "NR"),
        ),
        expected_operation="none",
    ),
    RobustnessCase(
        case_id="E04",
        category="multi_value_unsupported",
        question="Give the population aged 20 and 21 in Belgium in 2024.",
        expected_capability="known_unsupported",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(
            ("TIME_PERIOD", "2024"),
            ("age", "Y20"),
            ("age", "Y21"),
            ("freq", "A"),
            ("geo", "BE"),
            ("sex", "T"),
            ("unit", "NR"),
        ),
        expected_operation="none",
    ),
    RobustnessCase(
        case_id="E05",
        category="multi_value_unsupported",
        question=(
            "Give the population of 20-year-old women in Belgium and the "
            "Netherlands in 2024."
        ),
        expected_capability="known_unsupported",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(
            ("TIME_PERIOD", "2024"),
            ("age", "Y20"),
            ("freq", "A"),
            ("geo", "BE"),
            ("geo", "NL"),
            ("sex", "F"),
            ("unit", "NR"),
        ),
        expected_operation="none",
    ),
    RobustnessCase(
        case_id="E06",
        category="multi_value_unsupported",
        question=(
            "Give the population of 20-year-old women in Belgium for 2022, "
            "2023, and 2024."
        ),
        expected_capability="known_unsupported",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(
            ("TIME_PERIOD", "2022"),
            ("TIME_PERIOD", "2023"),
            ("TIME_PERIOD", "2024"),
            ("age", "Y20"),
            ("freq", "A"),
            ("geo", "BE"),
            ("sex", "F"),
            ("unit", "NR"),
        ),
        expected_operation="none",
    ),
    # Category F — safe-failure / hallucination resistance.
    RobustnessCase(
        case_id="F01",
        category="safe_failure",
        question="What was the population of Atlantis in 2024?",
        expected_capability="safe_failure",
        expected_dataset_code="DEMO_PJAN",
        expected_operation="none",
    ),
    RobustnessCase(
        case_id="F02",
        category="safe_failure",
        question="What was Belgium's population by eye colour in 2024?",
        expected_capability="safe_failure",
        expected_dataset_code="DEMO_PJAN",
        expected_operation="none",
    ),
    RobustnessCase(
        case_id="F03",
        category="safe_failure",
        question="What is the square root of Belgium's population in 2024?",
        expected_capability="safe_failure",
        expected_dataset_code="DEMO_PJAN",
        expected_operation="square_root",
    ),
    RobustnessCase(
        case_id="F04",
        category="safe_failure",
        question="What was the rate in Belgium in 2024?",
        expected_capability="safe_failure",
    ),
    RobustnessCase(
        case_id="F05",
        category="safe_failure",
        question="How many unicorns were unemployed in Germany in 2024?",
        expected_capability="safe_failure",
    ),
    RobustnessCase(
        case_id="F06",
        category="safe_failure",
        question="What was the happiness score of dragons in Belgium in 2024?",
        expected_capability="safe_failure",
    ),
)


def select_robustness_cases(
    case_ids: tuple[str, ...] | None = None,
) -> tuple[RobustnessCase, ...]:
    """Select Stage 11 cases by stable case id while preserving requested order."""

    if case_ids is None:
        return ROBUSTNESS_BENCHMARK_CASES

    by_id = {case.case_id: case for case in ROBUSTNESS_BENCHMARK_CASES}
    unknown = tuple(case_id for case_id in case_ids if case_id not in by_id)
    if unknown:
        raise ValueError("Unknown robustness case id: " + ", ".join(unknown))

    return tuple(by_id[case_id] for case_id in case_ids)
