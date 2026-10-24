"""Run the two-class multiskill queueing experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from conformal_cl.paths import PROJECT_ROOT
from conformal_cl.provenance import write_run_manifest
from conformal_cl.workforce import (
    QueueExperimentConfig,
    estimate_queue_oracle,
    run_queue_experiment,
    summarize_queue_runs,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replications", type=int, default=30)
    parser.add_argument("--oracle-seed", type=int, default=92821)
    parser.add_argument("--oracle-simulations", type=int, default=4000)
    parser.add_argument(
        "--oracle-file",
        type=Path,
        help="reuse a JSON oracle produced by estimate_queue_oracle",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "corrected" / "workforce" / "queueing.json",
    )
    args = parser.parse_args()
    if args.replications <= 0 or args.oracle_simulations <= 0:
        parser.error("replication and oracle simulation counts must be positive")

    if args.oracle_file:
        oracle = json.loads(args.oracle_file.read_text())
        oracle_simulations = oracle["simulations"]
    else:
        oracle_simulations = 30 if args.smoke else args.oracle_simulations
        oracle = estimate_queue_oracle(oracle_simulations, seed=args.oracle_seed)
    runs = []
    for seed in range(args.seed, args.seed + args.replications):
        config = (
            QueueExperimentConfig.smoke(seed)
            if args.smoke
            else QueueExperimentConfig(seed=seed)
        )
        runs.append(run_queue_experiment(config, oracle))
    result = {
        "study": {
            "seeds": list(range(args.seed, args.seed + args.replications)),
            "smoke": args.smoke,
        },
        "oracle": oracle,
        "summary": summarize_queue_runs(runs),
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
        run_kind="workforce_queueing",
        parameters={
            **result["study"],
            "oracle_seed": oracle["seed"],
            "oracle_simulations": oracle_simulations,
        },
        inputs=[args.oracle_file] if args.oracle_file else [],
        artifacts=[output],
    )
    print(json.dumps(result["summary"], indent=2))
