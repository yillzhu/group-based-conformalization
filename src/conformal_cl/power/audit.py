"""Falsification-first audit for residual geography in secure dispatch."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from conformal_cl.order_stats import (
    InsufficientCalibrationError,
    conformal_order_statistic,
)

from .case39 import load_case39
from .candidate_rcc import run_candidate_rcc
from .simulation import (
    DCSecuritySimulator,
    sample_dispatches,
    sample_security_focused_dispatches,
)


@dataclass(frozen=True)
class PowerAuditConfig:
    seed: int = 0
    alpha: float = 0.10
    training_size: int = 30_000
    design_size: int = 15_000
    calibration_size: int = 15_000
    candidate_size: int = 200_000
    evaluation_size: int = 100_000
    model_iterations: int = 400
    model_leaf_size: int = 60
    partition_leaves: int = 6
    partition_leaf_size: int = 1_000
    outage_probability: float = 0.05
    full_domain_fraction: float = 0.50

    @classmethod
    def smoke(cls, seed: int = 0) -> "PowerAuditConfig":
        return cls(
            seed=seed,
            training_size=2_000,
            design_size=1_200,
            calibration_size=1_200,
            candidate_size=12_000,
            evaluation_size=5_000,
            model_iterations=60,
            model_leaf_size=30,
            partition_leaves=4,
            partition_leaf_size=150,
        )


def run_power_audit(config: PowerAuditConfig) -> dict:
    """Run an independent-split screen before investing in literal RCC.

    The regression-tree partition is deliberately a diagnostic, not the RCC
    partition algorithm.  Training fits the conditional quantile; design data
    chooses a fixed residual partition; calibration data estimates corrections;
    and fresh simulation evaluates selected decisions.
    """

    if not 0. < config.alpha < 1.:
        raise ValueError("alpha must lie in (0, 1)")
    if not 0. < config.full_domain_fraction <= 1.:
        raise ValueError("full_domain_fraction must lie in (0, 1]")
    case = load_case39()
    simulator = DCSecuritySimulator(
        case, outage_probability=config.outage_probability
    )
    streams = np.random.SeedSequence(config.seed).spawn(5)
    train_rng, design_rng, calibration_rng, candidate_rng, evaluation_rng = (
        np.random.default_rng(stream) for stream in streams
    )
    train_x, train_y = _dataset(
        case, simulator, config.training_size, train_rng, config.full_domain_fraction
    )
    design_x, design_y = _dataset(
        case, simulator, config.design_size, design_rng, config.full_domain_fraction
    )
    calibration_x, calibration_y = _dataset(
        case, simulator, config.calibration_size, calibration_rng,
        config.full_domain_fraction,
    )

    train_features = _features(case, simulator, train_x)
    design_features = _features(case, simulator, design_x)
    calibration_features = _features(case, simulator, calibration_x)
    model = _fit_quantile_model(train_features, train_y, config)
    design_prediction = model.predict(design_features)
    design_score = design_y - design_prediction
    partition = _fit_partition(design_x / case.p_max, design_score, config)

    calibration_prediction = model.predict(calibration_features)
    calibration_score = calibration_y - calibration_prediction
    global_correction = conformal_order_statistic(
        calibration_score, alpha=config.alpha
    )
    design_leaf = partition.apply(design_x / case.p_max)
    calibration_leaf = partition.apply(calibration_x / case.p_max)
    leaf_correction = {}
    fallback_leaves = set()
    for leaf in np.unique(calibration_leaf):
        try:
            leaf_correction[int(leaf)] = conformal_order_statistic(
                calibration_score[calibration_leaf == leaf], alpha=config.alpha
            )
        except InsufficientCalibrationError:
            leaf_correction[int(leaf)] = global_correction
            fallback_leaves.add(int(leaf))
    for leaf in np.unique(design_leaf):
        if int(leaf) not in leaf_correction:
            leaf_correction[int(leaf)] = global_correction
            fallback_leaves.add(int(leaf))

    candidate_x = sample_security_focused_dispatches(
        simulator, config.candidate_size, candidate_rng
    )
    candidate_prediction = model.predict(_features(case, simulator, candidate_x))
    candidate_leaf = partition.apply(candidate_x / case.p_max)
    candidate_local_correction = np.array(
        [leaf_correction[int(leaf)] for leaf in candidate_leaf]
    )
    methods = {}
    for name, correction in (
        ("raw_quantile", np.zeros(len(candidate_x))),
        ("global_conformal", np.full(len(candidate_x), global_correction)),
        ("local_diagnostic", candidate_local_correction),
    ):
        feasible = candidate_prediction + correction <= 0.
        if not np.any(feasible):
            methods[name] = {"status": "no_candidate"}
            continue
        eligible = np.flatnonzero(feasible)
        index = eligible[np.argmin(candidate_x[eligible] @ case.linear_cost)]
        decision = candidate_x[index]
        evaluation_x = np.broadcast_to(
            decision, (config.evaluation_size, len(decision))
        )
        response = simulator.sample_response(evaluation_x, evaluation_rng)
        methods[name] = {
            "status": "selected",
            "cost": float(decision @ case.linear_cost),
            "dispatch": decision.tolist(),
            "predicted_quantile": float(candidate_prediction[index]),
            "correction": float(correction[index]),
            "partition_leaf": int(candidate_leaf[index]),
            "estimated_violation_probability": float(np.mean(response > 0.)),
            "evaluation_standard_error": float(
                np.sqrt(np.mean(response > 0.) * np.mean(response <= 0.) / len(response))
            ),
            "response_quantile": float(np.quantile(response, 1. - config.alpha)),
        }

    rcc = run_candidate_rcc(
        design_x / case.p_max,
        design_score,
        calibration_x / case.p_max,
        calibration_score,
        candidate_x / case.p_max,
        candidate_prediction,
        candidate_x @ case.linear_cost,
        alpha=config.alpha,
    )
    if rcc.index is None:
        methods["candidate_rcc"] = {
            "status": rcc.status,
            "splits": rcc.splits,
            "terminal_regions": rcc.terminal_regions,
            "certified_safe_regions": rcc.certified_safe_regions,
            "uncertified_regions": rcc.uncertified_regions,
            "discarded_regions": rcc.discarded_regions,
        }
    else:
        index = rcc.index
        decision = candidate_x[index]
        response = simulator.sample_response(
            np.broadcast_to(decision, (config.evaluation_size, len(decision))),
            evaluation_rng,
        )
        risk = np.mean(response > 0.)
        methods["candidate_rcc"] = {
            "status": "selected",
            "cost": float(decision @ case.linear_cost),
            "dispatch": decision.tolist(),
            "predicted_quantile": float(candidate_prediction[index]),
            "correction": float(rcc.correction),
            "estimated_violation_probability": float(risk),
            "evaluation_standard_error": float(
                np.sqrt(risk * (1. - risk) / len(response))
            ),
            "response_quantile": float(np.quantile(response, 1. - config.alpha)),
            "splits": rcc.splits,
            "terminal_regions": rcc.terminal_regions,
            "certified_safe_regions": rcc.certified_safe_regions,
            "uncertified_regions": rcc.uncertified_regions,
            "discarded_regions": rcc.discarded_regions,
            "scope": (
                "finite candidate safe-region approximation to Chapter 4 RCC"
            ),
        }

    leaves = []
    for leaf in sorted(leaf_correction):
        design_mask = design_leaf == leaf
        calibration_mask = calibration_leaf == leaf
        leaves.append(
            {
                "leaf": leaf,
                "design_count": int(design_mask.sum()),
                "calibration_count": int(calibration_mask.sum()),
                "design_mean_score": float(design_score[design_mask].mean()),
                "calibration_correction": float(leaf_correction[leaf]),
                "used_global_fallback": leaf in fallback_leaves,
            }
        )
    raw_coverage = float(np.mean(calibration_y <= calibration_prediction))
    correction_values = np.array(list(leaf_correction.values()))
    return {
        "config": asdict(config),
        "case": {
            "name": "PGLib-OPF IEEE 39-bus (case39_epri)",
            "buses": len(case.loads),
            "generators": len(case.p_max),
            "branches": len(case.branch_x),
            "connected_single_line_outages": simulator.contingencies,
            "nominal_demand_mw": case.demand,
        },
        "model": {
            "kind": "histogram gradient-boosted quantile regression",
            "sampling": (
                "mixture of full dispatch domain and nominally secure polytope"
            ),
            "features": (
                "normalized dispatch, generator headroom, and nominal DC line "
                "utilizations"
            ),
            "calibration_raw_upper_coverage": raw_coverage,
            "global_correction": float(global_correction),
        },
        "partition": {
            "kind": "held-out residual regression tree (diagnostic, not RCC)",
            "leaves": leaves,
            "correction_range": float(correction_values.max() - correction_values.min()),
        },
        "methods": methods,
    }


def _dataset(case, simulator, size, rng, full_domain_fraction):
    broad_size = max(1, round(size * full_domain_fraction))
    focused_size = size - broad_size
    pieces = [sample_dispatches(case, broad_size, rng)]
    if focused_size:
        pieces.append(
            sample_security_focused_dispatches(simulator, focused_size, rng)
        )
    x = np.vstack(pieces)
    rng.shuffle(x)
    return x, simulator.sample_response(x, rng)


def _features(case, simulator, x):
    normalized_dispatch = x / case.p_max
    injection = x @ simulator.generator_map.T - case.loads
    signed_utilization = injection @ simulator.nominal_flow_matrix.T / case.branch_rate
    lower_headroom = normalized_dispatch
    upper_headroom = 1. - normalized_dispatch
    summaries = np.column_stack((
        np.min(lower_headroom, axis=1),
        np.min(upper_headroom, axis=1),
        np.max(np.abs(signed_utilization), axis=1),
    ))
    return np.column_stack((
        normalized_dispatch, upper_headroom, signed_utilization, summaries,
    ))


def _fit_quantile_model(x, y, config):
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as error:
        raise RuntimeError("the power audit requires scikit-learn") from error
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=1. - config.alpha,
        max_iter=config.model_iterations,
        learning_rate=.08,
        max_leaf_nodes=31,
        min_samples_leaf=config.model_leaf_size,
        l2_regularization=1e-3,
        random_state=config.seed,
    ).fit(x, y)


def _fit_partition(x, score, config):
    try:
        from sklearn.tree import DecisionTreeRegressor
    except ImportError as error:
        raise RuntimeError("the power audit requires scikit-learn") from error
    return DecisionTreeRegressor(
        max_leaf_nodes=config.partition_leaves,
        min_samples_leaf=config.partition_leaf_size,
        random_state=config.seed,
    ).fit(x, score)
