"""Validate and aggregate corrected quadratic result batches."""

import argparse
from pathlib import Path

from conformal_cl.paths import PROJECT_ROOT
from conformal_cl.quadratic.summary import (
    load_corrected_results,
    summarize_corrected_results,
)
from conformal_cl.provenance import write_run_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "corrected" / "quadratic" / "summary",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = load_corrected_results(args.inputs)
    table_4_1, table_4_2 = summarize_corrected_results(results)
    table_4_1_path = args.output_dir / "table_4_1.csv"
    table_4_2_path = args.output_dir / "table_4_2_recorded_times.csv"
    table_4_1.to_csv(table_4_1_path, index=False)
    table_4_2.to_csv(table_4_2_path, index=False)

    write_run_manifest(
        args.output_dir / "summary_manifest.json",
        run_kind="corrected_quadratic_summary",
        parameters={
            "feasibility_denominator": "all_true_feasible",
            "timing_population": "all_true_feasible",
            "input_rows": len(results),
        },
        inputs=args.inputs,
        artifacts=[table_4_1_path, table_4_2_path],
    )
    print(table_4_1.to_string(index=False))
    print()
    print(
        table_4_2.pivot(
            index="dim", columns="Method", values="Mean_Recorded_Seconds"
        ).to_string()
    )
