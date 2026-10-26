#!/usr/bin/env python3
"""Test learned conditional quantiles on the transparent capacity examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import numpy as np

from conformal_cl.transparent.capacity import CASES
from conformal_cl.transparent.learned import run_learned_replication


def summarize(runs):
    methods = {}
    for method in runs[0]["methods"]:
        selected = [r["methods"][method] for r in runs if r["methods"][method]["status"] == "selected"]
        costs = [r["capacity_cost"] for r in selected]
        risks = [r["exact_violation_probability"] for r in selected]
        methods[method] = {
            "selected": len(selected),
            "reliable": sum(r <= .1 + 1e-12 for r in risks),
            "mean_cost": float(np.mean(costs)) if costs else None,
            "mean_exact_risk": float(np.mean(risks)) if risks else None,
            "max_exact_risk": max(risks) if risks else None,
            "mean_selected_raw_upper_coverage": float(np.mean([
                r["selected_point_raw_upper_coverage"] for r in selected
            ])) if selected else None,
            "min_selected_raw_upper_coverage": min(
                r["selected_point_raw_upper_coverage"] for r in selected
            ) if selected else None,
            "mean_abs_selected_upper_quantile_error": float(np.mean([
                abs(r["selected_point_upper_quantile_error"]) for r in selected
            ])) if selected else None,
        }
    quality = {}
    for area in ("full_domain_quality", "near_optimum_quality"):
        values = [r[area] for r in runs]
        quality[area] = {
            "quality_gate_pass_count": sum(v["quality_gate_passed"] for v in values),
            "means": {key: float(np.mean([v[key] for v in values]))
                      for key in values[0] if key != "quality_gate_passed"},
            "max_relative_excess_pinball": max(v["relative_excess_pinball"] for v in values),
            "max_upper_cdf_mae": max(v["upper_cdf_mae"] for v in values),
        }
    pairs = [(r["methods"]["rcc"], r["methods"]["global_cqr"]) for r in runs
             if all(r["methods"][m]["status"] == "selected" for m in ("rcc", "global_cqr"))]
    differences = np.array([a["capacity_cost"] - b["capacity_cost"] for a, b in pairs])
    leaves, counts = np.unique([r["model"]["selected_leaves"] for r in runs], return_counts=True)
    return {
        "replications": len(runs), "methods": methods, **quality,
        "quality_gate_pass_both": sum(
            r["full_domain_quality"]["quality_gate_passed"]
            and r["near_optimum_quality"]["quality_gate_passed"] for r in runs
        ),
        "selected_leaves": {str(int(n)): int(k) for n, k in zip(leaves, counts)},
        "rcc_minus_global_cost": {
            "paired_runs": len(pairs),
            "mean": float(np.mean(differences)) if len(pairs) else None,
            "min": float(np.min(differences)) if len(pairs) else None,
            "max": float(np.max(differences)) if len(pairs) else None,
            "rcc_cheaper": int(np.sum(differences < -1e-9)),
            "rcc_costlier": int(np.sum(differences > 1e-9)),
            "ties": int(np.sum(np.abs(differences) <= 1e-9)),
            "rcc_cheaper_and_both_reliable": sum(
                a["capacity_cost"] < b["capacity_cost"] - 1e-9
                and max(a["exact_violation_probability"], b["exact_violation_probability"]) <= .1 + 1e-12
                for a, b in pairs
            ),
        },
    }


def run_study(*, training_sizes=(8000,), replications=100, seed=20261026,
              validation_size=4000, calibration_size=4000, evaluation_size=20000,
              include_runs=False):
    import sklearn

    if min(training_sizes) < 100 or min(validation_size, calibration_size) < 100 or replications < 1:
        raise ValueError("positive replications and data splits of at least 100 rows required")
    result = {
        "config": {
            "training_sizes": list(training_sizes), "replications": replications, "seed": seed,
            "validation_size": validation_size, "calibration_size": calibration_size,
            "evaluation_size": evaluation_size, "alpha": .1, "delta": .1, "max_splits": 64,
            "leaf_candidates": [1, 2, 3, 4, 6, 8, 12],
            "quality_gate": {"relative_excess_pinball_at_most": .1, "upper_cdf_mae_at_most": .025},
            "near_optimum_evaluation_interval": [1., 1.2],
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scikit_learn": sklearn.__version__},
        "comparisons": [],
    }
    for training_size in training_sizes:
        for index, case in enumerate(CASES):
            # Reconstructing the same seed for each training size keeps the
            # validation/calibration/evaluation datasets paired across sizes.
            runs = [run_learned_replication(
                case, np.random.SeedSequence([seed, index, replication]),
                training_size=training_size, validation_size=validation_size,
                calibration_size=calibration_size, evaluation_size=evaluation_size,
            ) for replication in range(replications)]
            entry = {"case": case.name, "training_size": training_size, **summarize(runs)}
            if include_runs:
                entry["runs"] = runs
            result["comparisons"].append(entry)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-sizes", type=int, nargs="+", default=[8000])
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-runs", action="store_true")
    args = parser.parse_args()
    result = run_study(training_sizes=args.training_sizes, replications=args.replications,
                       seed=args.seed, include_runs=args.include_runs)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    for entry in result["comparisons"]:
        print(entry["case"], "training rows:", entry["training_size"])
        for name, method in entry["methods"].items():
            print(" ", name, json.dumps(method))
        print("  quality gates full/near/both:",
              entry["full_domain_quality"]["quality_gate_pass_count"],
              entry["near_optimum_quality"]["quality_gate_pass_count"], entry["quality_gate_pass_both"])
        print("  paired:", json.dumps(entry["rcc_minus_global_cost"]))


if __name__ == "__main__":
    main()
