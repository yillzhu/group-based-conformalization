"""RCC and diagnostics for a genuinely learned conditional-quantile model."""

from __future__ import annotations

from collections import deque

import numpy as np

from conformal_cl.groups import Group
from conformal_cl.order_stats import conformal_order_statistic

from .capacity import _report
from .quantile_models import fit_quantiles


def solve_learned_rcc(model, calibration_x, calibration_y, *, alpha=.1, delta=.1, max_splits=64):
    """Original queue logic, exact 1D spline extrema, and unresolved fallback."""

    x = np.asarray(calibration_x, dtype=float)
    predicted = model.predict(x)
    scores = np.maximum(calibration_y - predicted[:, 1], predicted[:, 0] - calibration_y)
    root = Group()
    root.conformalize(x[:, None], scores, alpha=alpha, delta=delta)
    left, right = float(model.knots[0]), float(model.knots[-1])
    queue = deque([(left, right, root)]) if root.Xhat else deque()
    unresolved = [] if root.Xhat else [(left, right, root)]
    safe = []
    splits = discarded = 0
    while queue and splits < max_splits:
        left, right, group = queue.popleft()
        minimum, minimum_x, maximum, maximum_x = model.extrema(left, right, 1)
        lower_minimum = model.extrema(left, right, 0)[0]
        minimum += group.qhat
        maximum += group.qhat
        if maximum <= 1e-9:
            safe.append((left, right, group))
            continue
        if lower_minimum - group.qhat >= -1e-9:
            discarded += 1
            continue
        if abs(maximum_x - minimum_x) <= 1e-9:
            unresolved.append((left, right, group))
            continue
        if minimum <= -1e-9 and abs(maximum - minimum) > 1e-9:
            cut = minimum_x - minimum * (maximum_x - minimum_x) / (maximum - minimum)
        else:
            cut = (minimum_x + maximum_x) / 2.
        normal = float(np.sign(maximum_x - minimum_x))
        coefficient = np.array([normal, normal * cut])
        if tuple(coefficient) in group.bounds or cut <= left or cut >= right:
            unresolved.append((left, right, group))
            continue
        first, second = group.split(coefficient, x[:, None], scores, alpha=alpha, delta=delta)
        bounds = ((left, cut), (cut, right)) if normal < 0. else ((cut, right), (left, cut))
        for (a, b), child in zip(bounds, (first, second)):
            (queue if child.Xhat else unresolved).append((a, b, child))
        splits += 1
    unresolved.extend(queue)
    proposals = []
    for route, cells in (("safe", safe), ("unresolved", unresolved)):
        for a, b, group in cells:
            decision = a if route == "safe" else model.cheapest_feasible(a, b, group.qhat)
            if decision is not None:
                proposals.append((decision, route != "safe", group.qhat, a, b, route))
    diagnostics = {"splits": splits, "safe_regions": len(safe),
                   "unresolved_regions": len(unresolved), "discarded_regions": discarded}
    if not proposals:
        return None, diagnostics
    decision, _, correction, a, b, route = min(proposals)
    return float(decision), {**diagnostics, "correction": float(correction),
                             "interval": [a, b], "route": route}


def model_diagnostics(case, model, x, *, alpha=.1):
    """Oracle-based held-out diagnostics; none of these quantities select the model."""

    x = np.asarray(x)
    predicted = model.predict(x)
    mean, scale = 1. - x, case.scale(x)
    levels = (alpha / 2., 1. - alpha / 2.)
    oracle = np.column_stack([mean + (2. * q - 1.) * scale for q in levels])
    loss = np.zeros(len(x))
    for index, quantile in enumerate(levels):
        offset = predicted[:, index] - mean
        inside = np.clip(offset, -scale, scale)
        value = (quantile * (scale - inside) ** 2
                 + (1. - quantile) * (scale + inside) ** 2) / (4. * scale)
        value += np.where(offset < -scale, quantile * (-scale - offset), 0.)
        value += np.where(offset > scale, (1. - quantile) * (offset - scale), 0.)
        loss += value
    oracle_loss = sum(scale * q * (1. - q) for q in levels)
    cdf = np.clip((predicted - mean[:, None] + scale[:, None]) / (2. * scale[:, None]), 0., 1.)
    result = {
        "expected_pinball": float(loss.mean()),
        "oracle_expected_pinball": float(oracle_loss.mean()),
        "relative_excess_pinball": float(loss.mean() / oracle_loss.mean() - 1.),
        "lower_quantile_mae": float(np.abs(predicted[:, 0] - oracle[:, 0]).mean()),
        "upper_quantile_mae": float(np.abs(predicted[:, 1] - oracle[:, 1]).mean()),
        "upper_cdf_mae": float(np.abs(cdf[:, 1] - levels[1]).mean()),
        "raw_upper_coverage": float(cdf[:, 1].mean()),
        "raw_interval_coverage": float((cdf[:, 1] - cdf[:, 0]).mean()),
        "mean_predictor_rmse": float(np.sqrt(np.mean(
            (model.mean_coefficient[0] + model.mean_coefficient[1] * x - mean) ** 2
        ))),
    }
    # A declared diagnostic threshold, not a filter for reported replications.
    result["quality_gate_passed"] = bool(
        result["relative_excess_pinball"] <= .10 and result["upper_cdf_mae"] <= .025
    )
    return result


def run_learned_replication(case, stream, *, training_size=8000, validation_size=4000,
                            calibration_size=4000, evaluation_size=20000, alpha=.1,
                            fit_options=None, population_diagnostics=False,
                            scaled_baseline=False, refit_baseline=False):
    train_rng, validation_rng, calibration_rng, evaluation_rng = [
        np.random.default_rng(child) for child in stream.spawn(4)
    ]
    tx, te = case.draw(training_size, train_rng)
    vx, ve = case.draw(validation_size, validation_rng)
    cx, ce = case.draw(calibration_size, calibration_rng)
    model, metadata = fit_quantiles(tx, 1. - tx + te, vx, 1. - vx + ve, alpha=alpha,
                                   **(fit_options or {}))
    response = 1. - cx + ce
    predicted = model.predict(cx)
    scores = np.maximum(response - predicted[:, 1], predicted[:, 0] - response)
    correction = conformal_order_statistic(scores, alpha=alpha, level_adjustment=1. / len(cx))
    lower, upper = case.edges[0], case.edges[-1]
    rcc_decision, rcc_details = solve_learned_rcc(model, cx, response, alpha=alpha)
    methods = {
        "raw_quantile": _report(case, model.cheapest_feasible(lower, upper)),
        "global_cqr": _report(case, model.cheapest_feasible(lower, upper, correction), correction=correction),
        "rcc": _report(case, rcc_decision, **rcc_details),
    }
    if scaled_baseline:
        from .quantile_models import QuantileSpline

        widths = (predicted[:, 1] - predicted[:, 0]) / 2.
        if np.any(widths <= 0.):
            raise ValueError("scaled calibration requires positive learned interval widths")
        scaled_correction = conformal_order_statistic(
            scores / widths, alpha=alpha, level_adjustment=1. / len(cx))
        knot_widths = (model.upper - model.lower) / 2.
        adjusted = QuantileSpline(model.knots,
            model.lower - scaled_correction * knot_widths,
            model.upper + scaled_correction * knot_widths, model.mean_coefficient)
        methods["scaled_global_cqr"] = _report(
            case, adjusted.cheapest_feasible(lower, upper),
            normalized_correction=scaled_correction)
    refit_model = None
    if refit_baseline:
        # Same labeled-data budget: use half the original calibration labels
        # for refitting, and reserve the other half for independent calibration.
        split = len(cx) // 2
        refit_model, refit_metadata = fit_quantiles(
            np.r_[tx, cx[:split]], np.r_[1. - tx + te, response[:split]],
            vx, 1. - vx + ve, alpha=alpha, **(fit_options or {}))
        rp = refit_model.predict(cx[split:])
        rs = np.maximum(response[split:] - rp[:, 1], rp[:, 0] - response[split:])
        rq = conformal_order_statistic(rs, alpha=alpha, level_adjustment=1. / len(rs))
        methods["refit_global_cqr"] = _report(
            case, refit_model.cheapest_feasible(lower, upper, rq),
            correction=rq, refit_selected_leaves=refit_metadata["selected_leaves"])
    for name, method in methods.items():
        if method["status"] == "selected":
            point = method["capacity_cost"]
            diagnostic_model = refit_model if name == "refit_global_cqr" else model
            lo, hi = diagnostic_model.predict([point])[0]
            s, m = float(case.scale(point)), 1. - point
            method["selected_point_raw_upper_coverage"] = float(np.clip((hi - m + s) / (2. * s), 0., 1.))
            method["selected_point_upper_quantile_error"] = float(hi - (m + (1. - alpha) * s))
    evaluation_x = evaluation_rng.uniform(lower, upper, evaluation_size)
    local_x = evaluation_rng.uniform(1., 1.2, evaluation_size)
    result = {
        "model": metadata,
        "full_domain_quality": model_diagnostics(case, model, evaluation_x, alpha=alpha),
        "near_optimum_quality": model_diagnostics(case, model, local_x, alpha=alpha),
        "methods": methods,
    }
    if population_diagnostics:
        from .population import correction_decomposition

        result["correction_decomposition"] = correction_decomposition(
            case, model, correction, rcc_details, alpha=alpha)
    return result
