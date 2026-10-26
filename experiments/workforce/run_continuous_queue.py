"""Run the continuous-capacity QED call-center RCC experiment."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from conformal_cl.paths import PROJECT_ROOT
from conformal_cl.provenance import write_run_manifest
from conformal_cl.workforce.continuous_queue_experiment import (
    ContinuousQueueExperimentConfig,
    run_continuous_queue_experiment,
    summarize_continuous_queue_runs,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replications", type=int, default=1)
    parser.add_argument(
        "--buffered",
        action="store_true",
        help="calibrate at 5%% for the fixed 10%% operational risk target",
    )
    parser.add_argument(
        "--spline-knots",
        type=int,
        default=0,
        help="number of dedicated-staff hinge knots (0 gives the affine model)",
    )
    parser.add_argument(
        "--full-support-training",
        action="store_true",
        help="train throughout the operational domain instead of incumbent policies",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "corrected"
            / "workforce"
            / "continuous_queue_rcc.json"
        ),
    )
    args = parser.parse_args()
    if args.replications <= 0:
        parser.error("--replications must be positive")
    seeds = list(range(args.seed, args.seed + args.replications))
    runs = []
    for seed in seeds:
        config = (
            ContinuousQueueExperimentConfig.smoke(seed)
            if args.smoke
            else ContinuousQueueExperimentConfig(seed=seed)
        )
        if args.buffered:
            config = replace(config, calibration_alpha=0.05)
        if args.full_support_training:
            config = replace(
                config,
                training_dedicated_lower=config.dedicated_lower,
                training_dedicated_upper=config.dedicated_upper,
                training_flexible_lower=config.flexible_lower,
                training_flexible_upper=config.flexible_upper,
            )
        config = replace(config, dedicated_spline_knots=args.spline_knots)
        runs.append(run_continuous_queue_experiment(config))
    result = {
        "study": {
            "seeds": seeds,
            "smoke": args.smoke,
            "buffered": args.buffered,
            "spline_knots": args.spline_knots,
            "full_support_training": args.full_support_training,
        },
        "summary": summarize_continuous_queue_runs(runs),
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
        run_kind="continuous_queue_rcc",
        parameters=result["study"],
        inputs=[],
        artifacts=[output],
    )
    print(json.dumps(result["summary"], indent=2))
