from eurostat_agent.evaluation import BenchmarkCase

CORE_BENCHMARK_CASES = (
    BenchmarkCase(
        question=("What was the population of 20-year-old women in Belgium in 2024?"),
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(
            ("TIME_PERIOD", "2024"),
            ("age", "Y20"),
            ("freq", "A"),
            ("geo", "BE"),
            ("sex", "F"),
            ("unit", "NR"),
        ),
        expected_operation="none",
    ),
    BenchmarkCase(
        question=("How many women aged 20 lived in Belgium on 1 January 2024?"),
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(
            ("TIME_PERIOD", "2024"),
            ("age", "Y20"),
            ("freq", "A"),
            ("geo", "BE"),
            ("sex", "F"),
            ("unit", "NR"),
        ),
        expected_operation="none",
    ),
    BenchmarkCase(
        question=("What was the population of 20-year-old men in Germany in 2024?"),
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(
            ("TIME_PERIOD", "2024"),
            ("age", "Y20"),
            ("freq", "A"),
            ("geo", "DE"),
            ("sex", "M"),
            ("unit", "NR"),
        ),
        expected_operation="none",
    ),
    BenchmarkCase(
        question="What is the total population across 2023 and 2024?",
        expected_dataset_code="DEMO_PJAN",
        expected_filters=(),
        expected_operation="sum",
    ),
)
