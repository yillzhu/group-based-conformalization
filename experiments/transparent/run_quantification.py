#!/usr/bin/env python3
"""Quantify correction bias/noise and stress complex conditional quantiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import numpy as np

from conformal_cl.transparent.capacity import CASES, CapacityCase
from conformal_cl.transparent.learned import run_learned_replication
from run_learned import summarize


def complex_cases():
    """Fixed random regimes, including both aligned and interior quiet regions.

    Construction seed is separate from all training/calibration streams. These
    are synthetic learning stresses, not realistic application benchmarks.
    """

    rng = np.random.default_rng(993)
    far = tuple(float(v) for v in rng.choice([.05, .8], 64))
    return (
        CapacityCase("complex_aligned", (0., *np.linspace(1.2, 4., 65)), (.1, *far)),
        CapacityCase("complex_interior", (*np.linspace(0., .9, 21), *np.linspace(1.4, 4., 65)),
                     (*tuple(float(v) for v in rng.choice([.05, .8], 20)), .1, *far)),
    )


def summarize_corrections(runs):
    records = [r["correction_decomposition"] for r in runs
               if r["correction_decomposition"] is not None]
    return {"selected_cells": len(records), "means": {
        key: float(np.mean([r[key] for r in records])) for key in records[0]
    }} if records else {"selected_cells": 0}


def run_study(replications=100, seed=20261027):
    import sklearn

    if replications < 1:
        raise ValueError("replications must be positive")
    result = {"config": {"replications": replications, "seed": seed,
        "validation_size": 4000, "calibration_size": 4000, "evaluation_size": 20000,
        "alpha": .1, "delta": .1, "max_splits": 64,
        "simple_leaf_candidates": [1, 2, 3, 4, 6, 8, 12],
        "simple_min_samples_leaf": "max(40, ceil(0.01*n_train))",
        "model_selection": "independent validation lower-plus-upper pinball loss",
        "refit_calibration_fraction": .5,
        "complex_construction_seed": 993,
        "complex_leaf_candidates": [1, 2, 3, 4, 6, 8, 12, 24, 48, 96],
        "complex_min_samples_leaf": 40},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scikit_learn": sklearn.__version__}, "comparisons": []}
    settings = [(CASES[1], (100, 250, 500, 2000, 8000, 32000), {})]
    settings.extend((case, (500, 2000, 8000),
                     {"leaf_candidates": (1, 2, 3, 4, 6, 8, 12, 24, 48, 96),
                      "min_samples_leaf": 40}) for case in complex_cases())
    for index, (case, sizes, options) in enumerate(settings):
        for n in sizes:
            runs = [run_learned_replication(case,
                np.random.SeedSequence([seed, index, replication]),
                training_size=n, fit_options=options,
                population_diagnostics=True, scaled_baseline=True, refit_baseline=True)
                for replication in range(replications)]
            entry = {"case": case.name, "edges": list(case.edges), "scales": list(case.scales),
                     "training_size": n, **summarize(runs),
                     "correction_decomposition": summarize_corrections(runs)}
            # Retain paired measurements, not just aggregates or successful runs.
            entry["runs"] = [{key: value for key, value in run.items() if key != "model"}
                             | {"selected_leaves": run["model"]["selected_leaves"]} for run in runs]
            result["comparisons"].append(entry)
            print(case.name, n, "cost/risk/reliable", {
                name: (v["mean_cost"], v["mean_exact_risk"], v["reliable"])
                for name, v in entry["methods"].items()}, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_study(args.replications, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
