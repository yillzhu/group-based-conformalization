"""Run the cross-trained workforce-planning comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from conformal_cl.paths import PROJECT_ROOT
from conformal_cl.provenance import write_run_manifest
from conformal_cl.workforce import run_study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="first replication seed")
    parser.add_argument("--replications", type=int, default=1)
    parser.add_argument(
        "--negative-control",
        action="store_true",
        help="train over the full grid instead of historical high-staffing decisions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "corrected" / "workforce" / "result.json",
    )
    args = parser.parse_args()

    if args.replications <= 0:
        parser.error("--replications must be positive")
    result = run_study(
        range(args.seed, args.seed + args.replications),
        smoke=args.smoke,
        negative_control=args.negative_control,
    )
    output = args.output
    if not output.is_absolute():
        output = (PROJECT_ROOT / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        parser.error(f"output already exists: {output}")
    output.write_text(json.dumps(result, indent=2) + "\n")
    write_run_manifest(
        output.with_suffix(".manifest.json"),
        run_kind=(
            "workforce_negative_control"
            if args.negative_control
            else "workforce_stress_test"
        ),
        parameters=result["study"],
        inputs=[],
        artifacts=[output],
    )
    print(json.dumps(result["summary"], indent=2))
