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
    minimize_piecewise_constraint,
    run_continuous_rcc,
)
from .models import fit_mean, fit_quantile, pinball_loss


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
    dedicated_spline_knots: int = 0
    model_family: str = "spline"
    neural_hidden_units: int = 16
    neural_epochs: int = 750
    diagnostic_size: int = 10000
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
    train_rng, cal_rng, eval_rng, oracle_rng, diagnostic_rng = map(
        np.random.default_rng, seed_sequence.spawn(5)
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
    point_model, lower_model, upper_model = _fit_models(train_x, train_y, config)
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
        selected = _optimize_model_constraint(
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
    diagnostic_x, diagnostic_y = sample_continuous_queue_dataset(
        domain_lower,
        domain_upper,
        config.diagnostic_size,
        diagnostic_rng,
        queue_config,
    )
    diagnostic_lower = lower_model.predict(diagnostic_x)
    diagnostic_upper = upper_model.predict(diagnostic_x)
    return {
        "config": asdict(config),
        "queue_config": asdict(queue_config),
        "domain": {"lower": domain_lower.tolist(), "upper": domain_upper.tolist()},
        "model": {
            "family": config.model_family,
            "lower": _model_metadata(lower_model),
            "upper": _model_metadata(upper_model),
            "lower_quantile_coefficient": (
                lower_model.coefficient.tolist()
                if hasattr(lower_model, "coefficient")
                else None
            ),
            "upper_quantile_coefficient": (
                upper_model.coefficient.tolist()
                if hasattr(upper_model, "coefficient")
                else None
            ),
            "global_correction": float(q_global),
            "dedicated_spline_breakpoints": (
                lower_model.center[0]
                + lower_model.scale[0]
                * np.asarray(getattr(lower_model, "dedicated_knots", ()))
            ).tolist(),
        },
        "predictive_diagnostics": {
            "lower_pinball_loss": pinball_loss(
                diagnostic_y, diagnostic_lower, config.calibration_alpha
            ),
            "upper_pinball_loss": pinball_loss(
                diagnostic_y, diagnostic_upper, 1.0 - config.calibration_alpha
            ),
            "raw_interval_coverage": float(
                np.mean(
                    (diagnostic_lower <= diagnostic_y)
                    & (diagnostic_y <= diagnostic_upper)
                )
            ),
            "raw_interval_mean_width": float(
                np.mean(diagnostic_upper - diagnostic_lower)
            ),
            "quantile_crossing_rate": float(
                np.mean(diagnostic_lower > diagnostic_upper)
            ),
        },
        "methods": methods,
    }


def summarize_continuous_queue_runs(runs: list[dict]) -> dict:
    return summarize_continuous_runs(runs)


def _fit_models(train_x, train_y, config):
    if config.model_family == "spline":
        point = fit_mean(
            train_x,
            train_y,
            dedicated_spline_knots=config.dedicated_spline_knots,
        )
        lower = fit_quantile(
            train_x,
            train_y,
            config.calibration_alpha,
            iterations=config.quantile_iterations,
            dedicated_spline_knots=config.dedicated_spline_knots,
        )
        upper = fit_quantile(
            train_x,
            train_y,
            1.0 - config.calibration_alpha,
            iterations=config.quantile_iterations,
            dedicated_spline_knots=config.dedicated_spline_knots,
        )
        return point, lower, upper
    if config.model_family == "relu":
        if config.dedicated_spline_knots:
            raise ValueError("spline knots are incompatible with a ReLU model")
        from .neural_models import fit_relu_mean, fit_relu_quantiles

        point = fit_relu_mean(
            train_x,
            train_y,
            hidden_units=config.neural_hidden_units,
            epochs=config.neural_epochs,
            seed=config.seed + 100_000,
        )
        lower, upper = fit_relu_quantiles(
            train_x,
            train_y,
            config.calibration_alpha,
            1.0 - config.calibration_alpha,
            hidden_units=config.neural_hidden_units,
            epochs=config.neural_epochs,
            seed=config.seed + 200_000,
        )
        return point, lower, upper
    raise ValueError(f"unknown model_family: {config.model_family!r}")


def _model_metadata(model):
    if hasattr(model, "coefficient"):
        return {
            "coefficient": model.coefficient.tolist(),
            "center": model.center.tolist(),
            "scale": model.scale.tolist(),
        }
    return {
        "hidden_weight": model.hidden_weight.tolist(),
        "hidden_bias": model.hidden_bias.tolist(),
        "output_weight": model.output_weight.tolist(),
        "output_bias": model.output_bias,
        "center": model.center.tolist(),
        "scale": model.scale.tolist(),
    }


def _optimize_model_constraint(
    model,
    correction,
    domain_lower,
    domain_upper,
    objective,
):
    lo = np.asarray(domain_lower, dtype=float)
    hi = np.asarray(domain_upper, dtype=float)
    polygon = np.asarray(
        [[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]
    )
    return minimize_piecewise_constraint(model, correction, polygon, objective)
