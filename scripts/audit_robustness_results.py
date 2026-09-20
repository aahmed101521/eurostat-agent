"""Create a deterministic companion audit of saved Stage 11 evidence, offline."""

from __future__ import annotations

import argparse
from pathlib import Path

from eurostat_agent.robustness_audit import audit_saved_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/stage11_robustness.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/stage11_robustness_diagnostics.json"),
    )
    args = parser.parse_args()
    audit_saved_report(args.input, args.output)
    print(f"Companion audit written to {args.output}")
    print(f"Original evidence preserved at {args.input}")


if __name__ == "__main__":
    main()
