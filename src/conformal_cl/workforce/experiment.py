"""End-to-end comparison for cross-trained workforce planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .conformal import calibrate_partition, design_partition, global_correction
from .models import fit_mean, fit_quantile
from .problem import (
    sample_dataset,
    staffing_cost,
    staffing_grid,
    violation_probabilities,
)


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 0
    alpha: float = 0.1
    training_size: int = 8000
    design_size: int = 8000
    calibration_size: int = 10000
    evaluation_size: int = 30000
    max_splits: int = 6
    min_design_size: int = 100
    quantile_iterations: int = 2500
    training_min_dedicated: int = 10

    @classmethod
    def smoke(cls, seed: int = 0) -> "ExperimentConfig":
        return cls(
            seed=seed,
            training_size=1200,
            design_size=1500,
            calibration_size=2000,
            evaluation_size=5000,
            max_splits=4,
            min_design_size=60,
            quantile_iterations=800,
        )


def run_experiment(config: ExperimentConfig) -> dict:
    if not 0.0 < config.alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    seed_sequence = np.random.SeedSequence(config.seed)
    train_rng, design_rng, cal_rng, eval_rng = map(
        np.random.default_rng, seed_sequence.spawn(4)
    )
    candidates = staffing_grid()
    costs = staffing_cost(candidates)

    training_candidates = candidates[
        candidates[:, 0] >= config.training_min_dedicated
    ]
    if not len(training_candidates):
        raise ValueError("training_min_dedicated excludes the full staffing grid")
    train_x, train_y = sample_dataset(
        training_candidates, config.training_size, train_rng
    )
    point_model = fit_mean(train_x, train_y)
    upper_model = fit_quantile(
        train_x,
        train_y,
        1.0 - config.alpha,
        iterations=config.quantile_iterations,
    )
    candidate_point = point_model.predict(candidates)
    candidate_upper = upper_model.predict(candidates)

    design_x, design_y = sample_dataset(candidates, config.design_size, design_rng)
    design_scores = design_y - upper_model.predict(design_x)
    cells = design_partition(
        design_x,
        design_scores,
        candidates,
        candidate_upper,
        alpha=config.alpha,
        max_splits=config.max_splits,
        min_design_size=config.min_design_size,
    )

    calibration_x, calibration_y = sample_dataset(
        candidates, config.calibration_size, cal_rng
    )
    calibration_scores = calibration_y - upper_model.predict(calibration_x)
    q_global = global_correction(calibration_scores, config.alpha)
    partition = calibrate_partition(
        cells, calibration_x, calibration_scores, alpha=config.alpha
    )

    violation = violation_probabilities(candidates, config.evaluation_size, eval_rng)
    methods = {
        "point_pto": candidate_point,
        "global_conformal_pto": candidate_upper + q_global,
        "group_based": candidate_upper + partition.correction(candidates),
        "oracle": violation - config.alpha,
    }
    results = {}
    for name, feasibility_measure in methods.items():
        feasible = feasibility_measure <= 0.0
        if not np.any(feasible):
            results[name] = {"status": "no_solution"}
            continue
        feasible_indices = np.flatnonzero(feasible)
        index = feasible_indices[np.argmin(costs[feasible_indices])]
        results[name] = {
            "status": "selected",
            "dedicated_per_class": int(candidates[index, 0]),
            "flexible": int(candidates[index, 1]),
            "cost": float(costs[index]),
            "estimated_violation_probability": float(violation[index]),
            "empirically_reliable": bool(violation[index] <= config.alpha),
            "fitted_feasibility_margin": float(feasibility_measure[index]),
        }

    return {
        "config": asdict(config),
        "model": {
            "target": "one-sided conditional response quantile",
            "quantile": 1.0 - config.alpha,
            "point_coefficient": point_model.coefficient.tolist(),
            "upper_quantile_coefficient": upper_model.coefficient.tolist(),
            "feature_center": upper_model.center.tolist(),
            "feature_scale": upper_model.scale.tolist(),
            "global_correction": float(q_global),
            "partition_cells": len(cells),
            "partition": [
                {
                    "lower": cell.lower.tolist(),
                    "upper": cell.upper.tolist(),
                    "design_count": int(cell.contains(design_x).sum()),
                    "calibration_count": int(count),
                    "correction": float(correction),
                }
                for cell, count, correction in zip(
                    partition.cells,
                    partition.calibration_counts,
                    partition.corrections,
                    strict=True,
                )
            ],
        },
        "methods": results,
    }
