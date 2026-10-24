"""Conformal optimization experiment driven by the multiskill queue simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .conformal import (
    calibrate_partition,
    calibrate_prediction_strata,
    design_partition,
    design_oblique_partition,
    design_optimization_partition,
    global_correction,
)
from .models import fit_mean, fit_quantile
from .problem import staffing_cost
from .queueing import QueueConfig, queue_violation_probabilities, sample_queue_responses


def queue_staffing_grid() -> np.ndarray:
    return np.asarray(
        [(d, f) for d in range(1, 9) for f in range(0, 8)], dtype=float
    )


@dataclass(frozen=True)
class QueueExperimentConfig:
    seed: int = 0
    alpha: float = 0.1
    training_size: int = 4000
    design_size: int = 4000
    calibration_size: int = 6000
    max_splits: int = 6
    min_design_size: int = 80
    quantile_iterations: int = 2500
    training_min_dedicated: int = 7
    decision_min_dedicated: int = 1
    prediction_strata: int = 8
    quadratic_model: bool = True

    @classmethod
    def smoke(cls, seed: int = 0) -> "QueueExperimentConfig":
        return cls(
            seed=seed,
            training_size=800,
            design_size=1000,
            calibration_size=1500,
            max_splits=4,
            min_design_size=40,
            quantile_iterations=800,
        )


def estimate_queue_oracle(
    simulations: int,
    *,
    seed: int = 92821,
    queue_config: QueueConfig | None = None,
) -> dict:
    """Estimate the fixed decision grid once using common random numbers."""

    queue_config = queue_config or QueueConfig()
    candidates = queue_staffing_grid()
    probabilities = queue_violation_probabilities(
        candidates, simulations, np.random.default_rng(seed), queue_config
    )
    return {
        "seed": seed,
        "simulations": simulations,
        "queue_config": asdict(queue_config),
        "candidates": candidates.astype(int).tolist(),
        "violation_probabilities": probabilities.tolist(),
    }


def run_queue_experiment(
    config: QueueExperimentConfig,
    oracle: dict,
    *,
    queue_config: QueueConfig | None = None,
) -> dict:
    """Fit, independently partition/calibrate, optimize, and evaluate one seed."""

    queue_config = queue_config or QueueConfig()
    if oracle["queue_config"] != asdict(queue_config):
        raise ValueError("oracle queue configuration does not match the experiment")
    candidates = queue_staffing_grid()
    if oracle["candidates"] != candidates.astype(int).tolist():
        raise ValueError("oracle candidate grid does not match the experiment")
    violation = np.asarray(oracle["violation_probabilities"], dtype=float)
    if len(violation) != len(candidates):
        raise ValueError("oracle has the wrong number of probabilities")

    seed_sequence = np.random.SeedSequence(config.seed)
    train_rng, design_rng, cal_rng = map(
        np.random.default_rng, seed_sequence.spawn(3)
    )
    training_candidates = candidates[
        candidates[:, 0] >= config.training_min_dedicated
    ]
    if len(np.unique(training_candidates[:, 0])) < 2:
        raise ValueError("training cutoff leaves too little dedicated-staff variation")

    train_x, train_y = _sample_dataset(
        training_candidates, config.training_size, train_rng, queue_config
    )
    point_model = fit_mean(train_x, train_y, quadratic=config.quadratic_model)
    upper_model = fit_quantile(
        train_x,
        train_y,
        1.0 - config.alpha,
        iterations=config.quantile_iterations,
        quadratic=config.quadratic_model,
    )
    lower_model = fit_quantile(
        train_x,
        train_y,
        config.alpha,
        iterations=config.quantile_iterations,
        quadratic=config.quadratic_model,
    )
    candidate_point = point_model.predict(candidates)
    candidate_lower = lower_model.predict(candidates)
    candidate_upper = upper_model.predict(candidates)

    design_x, design_y = _sample_dataset(
        candidates, config.design_size, design_rng, queue_config
    )
    design_scores = design_y - upper_model.predict(design_x)
    design_cqr_scores = np.maximum(
        design_y - upper_model.predict(design_x),
        lower_model.predict(design_x) - design_y,
    )
    refit_upper_model = fit_quantile(
        np.vstack((train_x, design_x)),
        np.concatenate((train_y, design_y)),
        1.0 - config.alpha,
        iterations=config.quantile_iterations,
        quadratic=config.quadratic_model,
    )
    candidate_refit_upper = refit_upper_model.predict(candidates)
    cells = design_partition(
        design_x,
        design_scores,
        candidates,
        candidate_upper,
        alpha=config.alpha,
        max_splits=config.max_splits,
        min_design_size=config.min_design_size,
    )
    oblique_cells = design_oblique_partition(
        design_x,
        design_scores,
        candidates,
        candidate_upper,
        alpha=config.alpha,
        max_splits=config.max_splits,
        min_design_size=config.min_design_size,
    )
    optimization_cells = design_optimization_partition(
        design_x,
        design_cqr_scores,
        candidates,
        candidate_lower,
        candidate_upper,
        alpha=config.alpha,
        max_splits=config.max_splits,
        min_design_size=config.min_design_size,
    )

    calibration_x, calibration_y = _sample_dataset(
        candidates, config.calibration_size, cal_rng, queue_config
    )
    calibration_scores = calibration_y - upper_model.predict(calibration_x)
    calibration_cqr_scores = np.maximum(
        calibration_y - upper_model.predict(calibration_x),
        lower_model.predict(calibration_x) - calibration_y,
    )
    refit_calibration_scores = (
        calibration_y - refit_upper_model.predict(calibration_x)
    )
    q_global = global_correction(calibration_scores, config.alpha)
    q_global_cqr = global_correction(calibration_cqr_scores, config.alpha)
    q_refit_global = global_correction(refit_calibration_scores, config.alpha)
    calibration_in_domain = (
        calibration_x[:, 0] >= config.decision_min_dedicated
    )
    q_domain = global_correction(
        calibration_scores[calibration_in_domain], config.alpha
    )
    partition = calibrate_partition(
        cells, calibration_x, calibration_scores, alpha=config.alpha
    )
    oblique_partition = calibrate_partition(
        oblique_cells, calibration_x, calibration_scores, alpha=config.alpha
    )
    optimization_partition = calibrate_partition(
        optimization_cells, calibration_x, calibration_cqr_scores, alpha=config.alpha
    )
    risk_strata = calibrate_prediction_strata(
        candidate_upper,
        upper_model.predict(calibration_x),
        calibration_scores,
        alpha=config.alpha,
        strata=config.prediction_strata,
    )

    costs = staffing_cost(candidates)
    eligible = candidates[:, 0] >= config.decision_min_dedicated
    methods = {
        "point_pto": candidate_point,
        "global_conformal_pto": candidate_upper + q_global,
        "global_cqr_pto": candidate_upper + q_global_cqr,
        "refit_global_conformal_pto": candidate_refit_upper + q_refit_global,
        "domain_conformal_pto": candidate_upper + q_domain,
        "group_based": candidate_upper + partition.correction(candidates),
        "oblique_group_based": candidate_upper
        + oblique_partition.correction(candidates),
        "rcc_group_based": candidate_upper
        + optimization_partition.correction(candidates),
        "risk_stratified": candidate_upper
        + risk_strata.correction(candidate_upper),
        "oracle": violation - config.alpha,
    }
    results = {
        name: _select(
            candidates, costs, violation, measure, config.alpha, eligible
        )
        for name, measure in methods.items()
    }
    _add_selected_cell_diagnostic(
        results["rcc_group_based"],
        optimization_partition,
        candidates,
        violation,
    )
    return {
        "config": asdict(config),
        "oracle": {"seed": oracle["seed"], "simulations": oracle["simulations"]},
        "model": {
            "target": "worst-class daily service-level shortfall",
            "point_coefficient": point_model.coefficient.tolist(),
            "lower_quantile_coefficient": lower_model.coefficient.tolist(),
            "upper_quantile_coefficient": upper_model.coefficient.tolist(),
            "refit_upper_quantile_coefficient": (
                refit_upper_model.coefficient.tolist()
            ),
            "feature_center": upper_model.center.tolist(),
            "feature_scale": upper_model.scale.tolist(),
            "global_correction": float(q_global),
            "global_cqr_correction": float(q_global_cqr),
            "refit_global_correction": float(q_refit_global),
            "domain_correction": float(q_domain),
            "domain_calibration_count": int(calibration_in_domain.sum()),
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
            "oblique_partition": [
                {
                    "bounds": [list(bound) for bound in cell.bounds],
                    "design_count": int(cell.contains(design_x).sum()),
                    "calibration_count": int(count),
                    "correction": float(correction),
                }
                for cell, count, correction in zip(
                    oblique_partition.cells,
                    oblique_partition.calibration_counts,
                    oblique_partition.corrections,
                    strict=True,
                )
            ],
            "optimization_partition": [
                {
                    "bounds": [list(bound) for bound in cell.bounds],
                    "design_count": int(cell.contains(design_x).sum()),
                    "calibration_count": int(count),
                    "correction": float(correction),
                }
                for cell, count, correction in zip(
                    optimization_partition.cells,
                    optimization_partition.calibration_counts,
                    optimization_partition.corrections,
                    strict=True,
                )
            ],
            "risk_strata": {
                "thresholds": risk_strata.thresholds.tolist(),
                "corrections": risk_strata.corrections.tolist(),
                "calibration_counts": risk_strata.calibration_counts.tolist(),
            },
        },
        "methods": results,
    }


def summarize_queue_runs(runs: list[dict]) -> dict:
    if not runs:
        raise ValueError("runs must not be empty")
    methods = tuple(runs[0]["methods"])
    summary = {"replications": len(runs), "methods": {}}
    for method in methods:
        results = [run["methods"][method] for run in runs]
        selected = [result for result in results if result["status"] == "selected"]
        reliable = [result for result in selected if result["empirically_reliable"]]
        method_summary = {
            "selection_rate": len(selected) / len(runs),
            "reliable_selection_rate": len(reliable) / len(runs),
            "unsafe_selection_count": len(selected) - len(reliable),
        }
        if selected:
            method_summary["mean_cost_when_selected"] = float(
                np.mean([result["cost"] for result in selected])
            )
            method_summary["mean_violation_when_selected"] = float(
                np.mean(
                    [result["estimated_violation_probability"] for result in selected]
                )
            )
            selected_cells = [
                result["selected_cell"]
                for result in selected
                if "selected_cell" in result
            ]
            if len(selected_cells) == len(selected):
                method_summary["mean_selected_cell_uniform_violation"] = float(
                    np.mean(
                        [
                            cell["uniform_mean_violation_probability"]
                            for cell in selected_cells
                        ]
                    )
                )
                method_summary["mean_optimizer_selection_gap"] = float(
                    np.mean(
                        [
                            cell["selected_violation_probability"]
                            - cell["uniform_mean_violation_probability"]
                            for cell in selected_cells
                        ]
                    )
                )
        summary["methods"][method] = method_summary
    return summary


def _sample_dataset(candidates, size, rng, queue_config):
    if size <= 0:
        raise ValueError("sample sizes must be positive")
    x = candidates[rng.integers(0, len(candidates), size=size)]
    return x, sample_queue_responses(x, rng, queue_config)


def _select(candidates, costs, violation, measure, alpha, eligible):
    feasible = (np.asarray(measure) <= 0.0) & np.asarray(eligible, dtype=bool)
    if not np.any(feasible):
        return {"status": "no_solution"}
    indices = np.flatnonzero(feasible)
    index = indices[np.argmin(costs[indices])]
    return {
        "status": "selected",
        "dedicated_per_class": int(candidates[index, 0]),
        "flexible": int(candidates[index, 1]),
        "cost": float(costs[index]),
        "estimated_violation_probability": float(violation[index]),
        "empirically_reliable": bool(violation[index] <= alpha),
        "fitted_feasibility_margin": float(measure[index]),
    }


def _add_selected_cell_diagnostic(
    result: dict,
    partition,
    candidates: np.ndarray,
    violation: np.ndarray,
) -> None:
    """Contrast optimizer risk with uniform risk over its RCC cell."""

    if result["status"] != "selected":
        return
    selected = np.asarray(
        [result["dedicated_per_class"], result["flexible"]], dtype=float
    )
    selected_index = int(np.flatnonzero(np.all(candidates == selected, axis=1))[0])
    containing = [
        index
        for index, cell in enumerate(partition.cells)
        if bool(cell.contains(candidates[[selected_index]])[0])
    ]
    if len(containing) != 1:
        raise RuntimeError("selected decision must belong to exactly one RCC cell")
    cell_index = containing[0]
    member = partition.cells[cell_index].contains(candidates)
    cell_violation = violation[member]
    result["selected_cell"] = {
        "index": cell_index,
        "candidate_count": int(member.sum()),
        "uniform_mean_violation_probability": float(np.mean(cell_violation)),
        "minimum_violation_probability": float(np.min(cell_violation)),
        "maximum_violation_probability": float(np.max(cell_violation)),
        "selected_violation_probability": float(violation[selected_index]),
    }
