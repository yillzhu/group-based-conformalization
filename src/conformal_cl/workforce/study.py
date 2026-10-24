"""Replication and summary utilities for the workforce experiment."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .experiment import ExperimentConfig, run_experiment


METHODS = ("point_pto", "global_conformal_pto", "group_based", "oracle")


def run_study(
    seeds,
    *,
    smoke: bool = False,
    negative_control: bool = False,
) -> dict:
    """Run independent replications and return raw results plus a summary."""

    seeds = [int(seed) for seed in seeds]
    if not seeds:
        raise ValueError("at least one seed is required")
    runs = []
    for seed in seeds:
        config = (
            ExperimentConfig.smoke(seed=seed)
            if smoke
            else ExperimentConfig(seed=seed)
        )
        if negative_control:
            config = replace(config, training_min_dedicated=4)
        runs.append(run_experiment(config))
    return {
        "study": {
            "seeds": seeds,
            "smoke": smoke,
            "negative_control": negative_control,
        },
        "summary": summarize_runs(runs),
        "runs": runs,
    }


def summarize_runs(runs: list[dict]) -> dict:
    if not runs:
        raise ValueError("runs must not be empty")
    alpha_values = {run["config"]["alpha"] for run in runs}
    if len(alpha_values) != 1:
        raise ValueError("all runs must use the same alpha")
    alpha = alpha_values.pop()
    summary = {"replications": len(runs), "alpha": alpha, "methods": {}}
    for method in METHODS:
        selected = [
            run["methods"][method]
            for run in runs
            if run["methods"][method]["status"] == "selected"
        ]
        if not selected:
            summary["methods"][method] = {"selection_rate": 0.0}
            continue
        costs = np.asarray([result["cost"] for result in selected])
        violations = np.asarray(
            [result["estimated_violation_probability"] for result in selected]
        )
        summary["methods"][method] = {
            "selection_rate": len(selected) / len(runs),
            "mean_cost": float(costs.mean()),
            "mean_estimated_violation_probability": float(violations.mean()),
            "empirical_reliability_rate": float(np.mean(violations <= alpha)),
        }

    global_cost = np.asarray(
        [run["methods"]["global_conformal_pto"].get("cost", np.nan) for run in runs]
    )
    group_cost = np.asarray(
        [run["methods"]["group_based"].get("cost", np.nan) for run in runs]
    )
    comparable = np.isfinite(global_cost) & np.isfinite(group_cost)
    summary["group_vs_global"] = {
        "comparable_runs": int(comparable.sum()),
        "group_cheaper_rate": float(np.mean(group_cost[comparable] < global_cost[comparable])),
        "equal_cost_rate": float(np.mean(group_cost[comparable] == global_cost[comparable])),
        "group_costlier_rate": float(np.mean(group_cost[comparable] > global_cost[comparable])),
    }
    return summary
