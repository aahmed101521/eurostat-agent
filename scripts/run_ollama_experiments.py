import argparse
from pathlib import Path

from eurostat_agent.benchmark import (
    CORE_BENCHMARK_CASES,
)
from eurostat_agent.ollama_experiments import (
    OLLAMA_EXPERIMENT_CONFIGS,
    run_ollama_experiment_suite,
    select_benchmark_cases,
    select_ollama_configs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Run local Ollama experiments against the Eurostat benchmark.")
    )

    parser.add_argument(
        "--config",
        action="append",
        help=(
            "Experiment configuration name. "
            "May be supplied more than once. "
            "Defaults to all configurations."
        ),
    )

    parser.add_argument(
        "--case",
        type=int,
        action="append",
        help=(
            "One-based benchmark case number. "
            "May be supplied more than once. "
            "Defaults to all cases."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/stage9_ollama_experiments.json"),
        help="Path for the JSON report.",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help=("List experiment configurations and benchmark cases, then exit."),
    )

    return parser


def print_available() -> None:
    print("Experiment configurations")
    print("=" * 72)

    for config in OLLAMA_EXPERIMENT_CONFIGS:
        print(f"{config.name:<28} {config.model:<20} {config.prompt_variant}")

    print()
    print("Benchmark cases")
    print("=" * 72)

    for index, case in enumerate(
        CORE_BENCHMARK_CASES,
        start=1,
    ):
        print(f"{index}. {case.question}")


def print_summary(
    comparison,
) -> None:
    print()
    print("Stage 9 Ollama experiment results")
    print("=" * 72)

    for result in comparison.results:
        summary = result.benchmark_run.summary

        print()
        print(result.config.name)
        print(f"  model:              {result.config.model}")
        print(f"  prompt:             {result.config.prompt_variant}")
        print(f"  completion_rate:    {summary.completion_rate:.3f}")
        print(f"  dataset_accuracy:   {summary.dataset_accuracy:.3f}")
        print(f"  filters_accuracy:   {summary.filters_accuracy:.3f}")
        print(f"  operation_accuracy: {summary.operation_accuracy:.3f}")
        print(f"  exact_match:        {summary.exact_match_accuracy:.3f}")

    if comparison.failures:
        print()
        print("Experiment failures")
        print("=" * 72)

        for failure in comparison.failures:
            print(
                f"{failure.config.name}: "
                f"{failure.stage}: "
                f"{failure.error_type}: "
                f"{failure.error_message}"
            )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        print_available()
        return

    config_names = tuple(args.config) if args.config else None

    case_indices = tuple(args.case) if args.case else None

    try:
        configs = select_ollama_configs(config_names)

        cases = select_benchmark_cases(case_indices)
    except ValueError as exc:
        parser.error(str(exc))

    comparison = run_ollama_experiment_suite(
        output_path=args.output,
        configs=configs,
        cases=cases,
        progress=lambda message: print(
            message,
            flush=True,
        ),
    )

    print_summary(comparison)


if __name__ == "__main__":
    main()
