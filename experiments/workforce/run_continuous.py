"""Run the continuous-FTE call-center RCC experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from conformal_cl.paths import PROJECT_ROOT
from conformal_cl.provenance import write_run_manifest
from conformal_cl.workforce.continuous_experiment import (
    ContinuousExperimentConfig,
    run_continuous_experiment,
    summarize_continuous_runs,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replications", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "corrected"
            / "workforce"
            / "continuous_rcc.json"
        ),
    )
    args = parser.parse_args()
    if args.replications <= 0:
        parser.error("--replications must be positive")
    seeds = list(range(args.seed, args.seed + args.replications))
    runs = [
        run_continuous_experiment(
            ContinuousExperimentConfig.smoke(seed)
            if args.smoke
            else ContinuousExperimentConfig(seed=seed)
        )
        for seed in seeds
    ]
    result = {
        "study": {"seeds": seeds, "smoke": args.smoke},
        "summary": summarize_continuous_runs(runs),
        "runs": runs,
    }
    output = args.output
    if not output.is_absolute():
        output = (PROJECT_ROOT / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        parser.error(f"output already exists: {output}")
    output.write_text(json.dumps(result, indent=2) + "\n")
    write_run_manifest(
        output.with_suffix(".manifest.json"),
        run_kind="continuous_workforce_rcc",
        parameters=result["study"],
        inputs=[],
        artifacts=[output],
    )
    print(json.dumps(result["summary"], indent=2))
