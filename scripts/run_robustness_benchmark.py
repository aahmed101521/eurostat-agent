import argparse
from pathlib import Path

from eurostat_agent.benchmark_cases import (
    ROBUSTNESS_BENCHMARK_CASES,
    select_robustness_cases,
)
from eurostat_agent.ollama_experiments import (
    OLLAMA_EXPERIMENT_CONFIGS,
    select_ollama_configs,
)
from eurostat_agent.ollama_robustness import (
    DEFAULT_ROBUSTNESS_CONFIG_NAMES,
    run_ollama_robustness_suite,
)
from eurostat_agent.robustness import RobustnessComparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Stage 11 end-to-end Eurostat robustness benchmark."
    )
    parser.add_argument(
        "--config",
        action="append",
        help=(
            "Experiment configuration name. May be supplied more than once. "
            "Defaults to the two constrained local-model configurations."
        ),
    )
    parser.add_argument(
        "--case",
        action="append",
        help=(
            "Stable Stage 11 case id such as A01 or F06. May be supplied more "
            "than once. Defaults to all 36 cases."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/stage11_robustness.json"),
        help="Path for the complete auditable JSON report.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available model configurations and Stage 11 cases, then exit.",
    )
    return parser


def print_available() -> None:
    print("Experiment configurations")
    print("=" * 88)
    for config in OLLAMA_EXPERIMENT_CONFIGS:
        marker = "*" if config.name in DEFAULT_ROBUSTNESS_CONFIG_NAMES else " "
        print(f"{marker} {config.name:<28} {config.model:<20} {config.prompt_variant}")

    print()
    print("Stage 11 robustness cases")
    print("=" * 88)
    for case in ROBUSTNESS_BENCHMARK_CASES:
        print(
            f"{case.case_id:<4} {case.category:<27} "
            f"{case.expected_capability:<18} {case.question}"
        )


def print_summary(comparison: RobustnessComparison) -> None:
    print()
    print("Stage 11 robustness results")
    print("=" * 88)

    for run in comparison.runs:
        summary = run.summary
        print()
        print(run.config.name)
        print(f"  model:                       {run.config.model}")
        print(f"  prompt:                      {run.config.prompt_variant}")
        print(f"  total cases:                 {summary.total_cases}")
        print(f"  completed cases:             {summary.completed_cases}")
        print(f"  supported cases:             {summary.supported_cases}")
        print(f"  completed supported:         {summary.completed_supported_cases}")
        print(f"  exact supported:             {summary.exact_supported_cases}")
        print(f"  safe failures:               {summary.safe_failures}")
        print(f"  architecture limitations:    {summary.architecture_limitation_count}")
        print(f"  unexpected failures:         {summary.unexpected_failure_count}")
        print(f"  provenance violations:       {summary.provenance_violation_count}")
        print(
            "  supported completion rate:   "
            f"{summary.supported_case_completion_rate:.3f}"
        )
        print(
            f"  supported exact-match rate:  {summary.supported_case_exact_match:.3f}"
        )
        print(f"  safe-failure rate:           {summary.safe_failure_rate:.3f}")
        print(f"  dataset accuracy:            {summary.dataset_accuracy:.3f}")
        print(f"  filters accuracy:            {summary.filters_accuracy:.3f}")
        print(f"  operation accuracy:          {summary.operation_accuracy:.3f}")
        print(f"  model calls:                 {summary.model_call_count}")
        print(
            "  cache hits / misses:         "
            f"{summary.cache_hits}/{summary.cache_misses}"
        )
        print(f"  median total latency:        {summary.median_total_latency_seconds}")
        print(f"  p95 total latency:           {summary.p95_total_latency_seconds}")
        print(
            "  temporal year/full-date:     "
            f"{summary.temporal_behavior.annual_year_values}/"
            f"{summary.temporal_behavior.full_date_values}"
        )
        print(
            f"  failure stages:              {dict(summary.failure_stage_distribution)}"
        )

    if comparison.failures:
        print()
        print("Configuration-level failures")
        print("=" * 88)
        for failure in comparison.failures:
            print(
                f"{failure.config.name}: {failure.error_type}: {failure.error_message}"
            )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        print_available()
        return

    config_names = (
        tuple(args.config) if args.config else DEFAULT_ROBUSTNESS_CONFIG_NAMES
    )
    case_ids = tuple(args.case) if args.case else None

    try:
        configs = select_ollama_configs(config_names)
        cases = select_robustness_cases(case_ids)
    except ValueError as exc:
        parser.error(str(exc))

    comparison = run_ollama_robustness_suite(
        output_path=args.output,
        configs=configs,
        cases=cases,
        progress=lambda message: print(message, flush=True),
    )
    print_summary(comparison)


if __name__ == "__main__":
    main()
