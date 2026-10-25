"""Repeat the fair-information oracle tuition benchmark across random seeds."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.runs <= 0 or args.jobs <= 0:
        parser.error("--runs and --jobs must be positive")

    benchmark = Path(__file__).with_name("oracle_benchmark.py")
    with tempfile.TemporaryDirectory(prefix="oracle-tuition-") as directory:
        directory = Path(directory)

        def run(seed):
            output = directory / f"seed_{seed}.json"
            subprocess.run(
                [
                    sys.executable,
                    str(benchmark),
                    "--seed",
                    str(seed),
                    "--output",
                    str(output),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            return json.loads(output.read_text())

        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            reports = list(executor.map(run, range(args.runs)))

    global_cost = [
        report["summary"]["global"]["cohort_weighted_mean_merit_dollars"]
        for report in reports
    ]
    local_cost = [
        report["summary"]["localized"]["cohort_weighted_mean_merit_dollars"]
        for report in reports
    ]
    savings = [global_value - local_value for global_value, local_value in zip(global_cost, local_cost)]
    result = {
        "runs": args.runs,
        "segment_decisions": 12 * args.runs,
        "mean_global_dollars": statistics.fmean(global_cost),
        "mean_localized_dollars": statistics.fmean(local_cost),
        "mean_localized_saving_dollars": statistics.fmean(savings),
        "median_localized_saving_dollars": statistics.median(savings),
        "minimum_localized_saving_dollars": min(savings),
        "maximum_localized_saving_dollars": max(savings),
        "localized_cheaper_runs": sum(value > 1e-8 for value in savings),
        "localized_costlier_runs": sum(value < -1e-8 for value in savings),
        "tied_runs": sum(abs(value) <= 1e-8 for value in savings),
        "global_oracle_predictive_violations": sum(
            report["summary"]["global"]["predictive_90pct_violations"]
            for report in reports
        ),
        "localized_oracle_predictive_violations": sum(
            report["summary"]["localized"]["predictive_90pct_violations"]
            for report in reports
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
