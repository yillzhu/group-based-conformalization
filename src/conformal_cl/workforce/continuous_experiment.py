"""Continuous-FTE call-center experiment using the literal RCC queue logic."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from conformal_cl.order_stats import conformal_order_statistic

from .continuous_rcc import (
    affine_parameters,
    clip_polygon,
    minimize_over_vertices,
    run_continuous_rcc,
)
from .models import fit_mean, fit_quantile
from .problem import (
    approximate_continuous_oracle,
    sample_continuous_dataset,
    violation_probabilities,
)


@dataclass(frozen=True)
class ContinuousExperimentConfig:
    seed: int = 0
    alpha: float = 0.1
    delta: float = 0.1
    training_size: int = 8000
    calibration_size: int = 10000
    evaluation_size: int = 30000
    oracle_size: int = 30000
    oracle_grid_size: int = 2001
    max_splits: int = 64
    quantile_iterations: int = 2500
    training_min_dedicated: float = 10.0
    dedicated_lower: float = 4.0
    dedicated_upper: float = 15.0
    flexible_lower: float = 0.0
    flexible_upper: float = 18.0

    @classmethod
    def smoke(cls, seed: int = 0) -> "ContinuousExperimentConfig":
        return cls(
            seed=seed,
            training_size=1200,
            calibration_size=2000,
            evaluation_size=5000,
            oracle_size=5000,
            oracle_grid_size=401,
            quantile_iterations=800,
            max_splits=24,
        )


def run_continuous_experiment(config: ContinuousExperimentConfig) -> dict:
    """Fit interval models, run literal continuous RCC, and evaluate decisions."""

    domain_lower = np.asarray(
        [config.dedicated_lower, config.flexible_lower], dtype=float
    )
    domain_upper = np.asarray(
        [config.dedicated_upper, config.flexible_upper], dtype=float
    )
    training_lower = domain_lower.copy()
    training_lower[0] = config.training_min_dedicated
    if np.any(training_lower >= domain_upper):
        raise ValueError("training domain must be nonempty")
    seed_sequence = np.random.SeedSequence(config.seed)
    train_rng, cal_rng, eval_rng, oracle_rng = map(
        np.random.default_rng, seed_sequence.spawn(4)
    )
    train_x, train_y = sample_continuous_dataset(
        training_lower, domain_upper, config.training_size, train_rng
    )
    calibration_x, calibration_y = sample_continuous_dataset(
        domain_lower, domain_upper, config.calibration_size, cal_rng
    )
    point_model = fit_mean(train_x, train_y)
    lower_model = fit_quantile(
        train_x,
        train_y,
        config.alpha,
        iterations=config.quantile_iterations,
    )
    upper_model = fit_quantile(
        train_x,
        train_y,
        1.0 - config.alpha,
        iterations=config.quantile_iterations,
    )
    lower_calibration = lower_model.predict(calibration_x)
    upper_calibration = upper_model.predict(calibration_x)
    scores = np.maximum(
        calibration_y - upper_calibration,
        lower_calibration - calibration_y,
    )
    q_global = conformal_order_statistic(
        scores,
        alpha=config.alpha,
        level_adjustment=1.0 / len(scores),
    )

    rcc = run_continuous_rcc(
        lower_model,
        upper_model,
        calibration_x,
        calibration_y,
        domain_lower=domain_lower,
        domain_upper=domain_upper,
        alpha=config.alpha,
        delta=config.delta,
        max_splits=config.max_splits,
    )
    decisions: dict[str, tuple[np.ndarray, float, dict]] = {}
    point = _optimize_affine_constraint(
        point_model, 0.0, domain_lower, domain_upper
    )
    if point is not None:
        decisions["point_pto"] = (*point, {})
    global_cqr = _optimize_affine_constraint(
        upper_model, q_global, domain_lower, domain_upper
    )
    if global_cqr is not None:
        decisions["global_cqr_pto"] = (*global_cqr, {})
    if rcc.decision is not None:
        decisions["rcc"] = (
            rcc.decision,
            float(rcc.objective),
            {
                "route": rcc.route,
                "correction": rcc.correction,
                "splits": rcc.splits,
                "safe_regions": rcc.safe_regions,
                "unresolved_regions": rcc.unresolved_regions,
                "discarded_regions": rcc.discarded_regions,
            },
        )
    oracle_decision, oracle_cost, oracle_design_violation = (
        approximate_continuous_oracle(
            domain_lower=domain_lower,
            domain_upper=domain_upper,
            alpha=config.alpha,
            simulations=config.oracle_size,
            rng=oracle_rng,
            dedicated_grid_size=config.oracle_grid_size,
        )
    )
    decisions["oracle"] = (
        oracle_decision,
        oracle_cost,
        {"design_sample_violation_probability": oracle_design_violation},
    )

    names = list(decisions)
    selected_x = np.vstack([decisions[name][0] for name in names])
    violations = violation_probabilities(
        selected_x, config.evaluation_size, eval_rng
    )
    methods = {
        name: {
            "status": "selected",
            "dedicated_fte_per_class": float(decisions[name][0][0]),
            "flexible_fte": float(decisions[name][0][1]),
            "cost": float(decisions[name][1]),
            "estimated_violation_probability": float(violations[index]),
            "empirically_reliable": bool(violations[index] <= config.alpha),
            **decisions[name][2],
        }
        for index, name in enumerate(names)
    }
    for name in ("point_pto", "global_cqr_pto", "rcc", "oracle"):
        methods.setdefault(name, {"status": "no_solution"})
    return {
        "config": asdict(config),
        "domain": {"lower": domain_lower.tolist(), "upper": domain_upper.tolist()},
        "model": {
            "lower_quantile_coefficient": lower_model.coefficient.tolist(),
            "upper_quantile_coefficient": upper_model.coefficient.tolist(),
            "global_correction": float(q_global),
        },
        "methods": methods,
    }


def summarize_continuous_runs(runs: list[dict]) -> dict:
    if not runs:
        raise ValueError("runs must not be empty")
    methods = ("point_pto", "global_cqr_pto", "rcc", "oracle")
    summary = {"replications": len(runs), "methods": {}}
    for method in methods:
        selected = [
            run["methods"][method]
            for run in runs
            if run["methods"][method]["status"] == "selected"
        ]
        method_summary = {"selection_rate": len(selected) / len(runs)}
        if selected:
            method_summary.update(
                {
                    "reliable_selection_rate": float(
                        np.mean(
                            [result["empirically_reliable"] for result in selected]
                        )
                    ),
                    "mean_cost_when_selected": float(
                        np.mean([result["cost"] for result in selected])
                    ),
                    "mean_violation_when_selected": float(
                        np.mean(
                            [
                                result["estimated_violation_probability"]
                                for result in selected
                            ]
                        )
                    ),
                }
            )
        summary["methods"][method] = method_summary
    paired = [
        (run["methods"]["rcc"], run["methods"]["global_cqr_pto"])
        for run in runs
        if run["methods"]["rcc"]["status"] == "selected"
        and run["methods"]["global_cqr_pto"]["status"] == "selected"
    ]
    differences = np.asarray(
        [rcc["cost"] - global_cqr["cost"] for rcc, global_cqr in paired]
    )
    summary["rcc_vs_global_cqr"] = {
        "comparable_runs": len(paired),
        "rcc_cheaper_count": int(np.sum(differences < -1e-9)),
        "equal_cost_count": int(np.sum(np.abs(differences) <= 1e-9)),
        "rcc_costlier_count": int(np.sum(differences > 1e-9)),
        "mean_paired_cost_difference": float(np.mean(differences)),
    }
    return summary


def _optimize_affine_constraint(model, correction, domain_lower, domain_upper):
    intercept, slope = affine_parameters(model)
    lo = np.asarray(domain_lower, dtype=float)
    hi = np.asarray(domain_upper, dtype=float)
    polygon = np.asarray(
        [[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]
    )
    feasible = clip_polygon(polygon, slope, -intercept - correction)
    return minimize_over_vertices(feasible, np.asarray([2.0, 2.4]))
