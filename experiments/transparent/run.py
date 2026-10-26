#!/usr/bin/env python3
"""Population and repeated-sample diagnostics with exact optimization and risk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import numpy as np

from conformal_cl.transparent.capacity import (
    CASES, empirical_comparison, population_comparison, solve_rcc,
)


def run_study(*, replications=200, calibration_size=4000, seed=0):
    if replications <= 0 or calibration_size < 100:
        raise ValueError("positive replications and at least 100 calibration rows required")
    case_streams = np.random.SeedSequence(seed).spawn(len(CASES))
    result = {
        "config": {"seed": seed, "replications": replications,
                   "calibration_size": calibration_size, "alpha": .1,
                   "beta": .05, "delta": .1},
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "population": [population_comparison(case) for case in CASES],
        "population_delta_sensitivity": [
            {"case": case.name, "delta": delta, "rcc": solve_rcc(case, delta=delta)}
            for case in CASES for delta in (.2, .1, .05, .02, .01)
        ],
        "finite_sample": [],
    }
    for case, stream in zip(CASES, case_streams):
        runs = [
            empirical_comparison(case, calibration_size, np.random.default_rng(child))
            for child in stream.spawn(replications)
        ]
        summary = {}
        for method in runs[0]:
            selected = [run[method] for run in runs if run[method]["status"] == "selected"]
            risks = [r["exact_violation_probability"] for r in selected]
            costs = [r["capacity_cost"] for r in selected]
            summary[method] = {
                "selected": len(selected),
                "reliable": sum(r <= .1 + 1e-12 for r in risks),
                "mean_cost": float(np.mean(costs)) if costs else None,
                "mean_exact_risk": float(np.mean(risks)) if risks else None,
                "max_exact_risk": max(risks) if risks else None,
            }
        result["finite_sample"].append({"case": case.name, "summary": summary, "runs": runs})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=200)
    parser.add_argument("--calibration-size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-runs", action="store_true", help="retain individual runs in the saved JSON")
    args = parser.parse_args()
    result = run_study(replications=args.replications, calibration_size=args.calibration_size, seed=args.seed)
    display = {**result, "finite_sample": [
        {key: value for key, value in case.items() if key != "runs"}
        for case in result["finite_sample"]
    ]}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        saved = result if args.include_runs else display
        args.output.write_text(json.dumps(saved, indent=2) + "\n")
    print(json.dumps(display, indent=2))


if __name__ == "__main__":
    main()
