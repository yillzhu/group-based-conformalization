"""Literal continuous RCC on a congestion-sensitive call-center approximation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from conformal_cl.order_stats import conformal_order_statistic

from .continuous_experiment import summarize_continuous_runs
from .continuous_queue import (
    ContinuousQueueConfig,
    approximate_continuous_queue_oracle,
    continuous_queue_violation_probabilities,
    sample_continuous_queue_dataset,
)
from .continuous_rcc import (
    affine_parameters,
    clip_polygon,
    minimize_over_vertices,
    run_continuous_rcc,
)
from .models import fit_mean, fit_quantile


@dataclass(frozen=True)
class ContinuousQueueExperimentConfig:
    seed: int = 0
    operational_alpha: float = 0.1
    calibration_alpha: float = 0.1
    delta: float = 0.1
    training_size: int = 8000
    calibration_size: int = 10000
    evaluation_size: int = 30000
    oracle_size: int = 30000
    oracle_grid_size: int = 2001
    max_splits: int = 64
    quantile_iterations: int = 2500
    dedicated_lower: float = 4.0
    dedicated_upper: float = 12.0
    flexible_lower: float = 0.0
    flexible_upper: float = 9.0
    training_dedicated_lower: float = 4.0
    training_dedicated_upper: float = 9.0
    training_flexible_lower: float = 1.0
    training_flexible_upper: float = 6.0

    @classmethod
    def smoke(cls, seed: int = 0) -> "ContinuousQueueExperimentConfig":
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


def run_continuous_queue_experiment(
    config: ContinuousQueueExperimentConfig,
    *,
    queue_config: ContinuousQueueConfig | None = None,
) -> dict:
    queue_config = queue_config or ContinuousQueueConfig()
    domain_lower = np.asarray(
        [config.dedicated_lower, config.flexible_lower], dtype=float
    )
    domain_upper = np.asarray(
        [config.dedicated_upper, config.flexible_upper], dtype=float
    )
    training_lower = np.asarray(
        [config.training_dedicated_lower, config.training_flexible_lower],
        dtype=float,
    )
    training_upper = np.asarray(
        [config.training_dedicated_upper, config.training_flexible_upper],
        dtype=float,
    )
    seed_sequence = np.random.SeedSequence(config.seed)
    train_rng, cal_rng, eval_rng, oracle_rng = map(
        np.random.default_rng, seed_sequence.spawn(4)
    )
    train_x, train_y = sample_continuous_queue_dataset(
        training_lower,
        training_upper,
        config.training_size,
        train_rng,
        queue_config,
    )
    calibration_x, calibration_y = sample_continuous_queue_dataset(
        domain_lower,
        domain_upper,
        config.calibration_size,
        cal_rng,
        queue_config,
    )
    point_model = fit_mean(train_x, train_y)
    lower_model = fit_quantile(
        train_x,
        train_y,
        config.calibration_alpha,
        iterations=config.quantile_iterations,
    )
    upper_model = fit_quantile(
        train_x,
        train_y,
        1.0 - config.calibration_alpha,
        iterations=config.quantile_iterations,
    )
    calibration_scores = np.maximum(
        calibration_y - upper_model.predict(calibration_x),
        lower_model.predict(calibration_x) - calibration_y,
    )
    q_global = conformal_order_statistic(
        calibration_scores,
        alpha=config.calibration_alpha,
        level_adjustment=1.0 / len(calibration_scores),
    )
    objective = np.asarray(
        [queue_config.dedicated_cost, queue_config.flexible_cost]
    )
    rcc = run_continuous_rcc(
        lower_model,
        upper_model,
        calibration_x,
        calibration_y,
        domain_lower=domain_lower,
        domain_upper=domain_upper,
        objective_slope=objective,
        alpha=config.calibration_alpha,
        delta=config.delta,
        max_splits=config.max_splits,
    )
    decisions = {}
    for name, model, correction in (
        ("point_pto", point_model, 0.0),
        ("global_cqr_pto", upper_model, q_global),
    ):
        selected = _optimize_affine_constraint(
            model,
            correction,
            domain_lower,
            domain_upper,
            objective,
        )
        if selected is not None:
            decisions[name] = (*selected, {})
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
    oracle_decision, oracle_cost, oracle_violation = (
        approximate_continuous_queue_oracle(
            domain_lower=domain_lower,
            domain_upper=domain_upper,
            alpha=config.operational_alpha,
            simulations=config.oracle_size,
            rng=oracle_rng,
            config=queue_config,
            dedicated_grid_size=config.oracle_grid_size,
        )
    )
    decisions["oracle"] = (
        oracle_decision,
        oracle_cost,
        {"design_sample_violation_probability": oracle_violation},
    )
    names = list(decisions)
    selected_x = np.vstack([decisions[name][0] for name in names])
    violations = continuous_queue_violation_probabilities(
        selected_x,
        config.evaluation_size,
        eval_rng,
        queue_config,
    )
    methods = {
        name: {
            "status": "selected",
            "dedicated_fte_per_class": float(decisions[name][0][0]),
            "flexible_fte": float(decisions[name][0][1]),
            "cost": float(decisions[name][1]),
            "estimated_violation_probability": float(violations[index]),
            "empirically_reliable": bool(
                violations[index] <= config.operational_alpha
            ),
            **decisions[name][2],
        }
        for index, name in enumerate(names)
    }
    for name in ("point_pto", "global_cqr_pto", "rcc", "oracle"):
        methods.setdefault(name, {"status": "no_solution"})
    return {
        "config": asdict(config),
        "queue_config": asdict(queue_config),
        "domain": {"lower": domain_lower.tolist(), "upper": domain_upper.tolist()},
        "model": {
            "lower_quantile_coefficient": lower_model.coefficient.tolist(),
            "upper_quantile_coefficient": upper_model.coefficient.tolist(),
            "global_correction": float(q_global),
        },
        "methods": methods,
    }


def summarize_continuous_queue_runs(runs: list[dict]) -> dict:
    return summarize_continuous_runs(runs)


def _optimize_affine_constraint(
    model,
    correction,
    domain_lower,
    domain_upper,
    objective,
):
    intercept, slope = affine_parameters(model)
    lo = np.asarray(domain_lower, dtype=float)
    hi = np.asarray(domain_upper, dtype=float)
    polygon = np.asarray(
        [[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]
    )
    feasible = clip_polygon(polygon, slope, -intercept - correction)
    return minimize_over_vertices(feasible, objective)
