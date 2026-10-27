#!/usr/bin/env python3
"""Learn the conditional lower tenth percentile directly, then optimize.

Model selection uses only the 240 training observations. An untouched set of
240 observations supplies a one-sided split-conformal correction. The known
response law is used only to evaluate decisions after they have been made.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import KFold

from conformal_cl.order_stats import conformal_order_statistic
from conformal_cl.transparent.learned import solve_learned_rcc
from conformal_cl.transparent.quantile_models import QuantileSpline


TARGET = .35
ALPHA = .1
KNOTS = np.linspace(0., 1., 6)
GRID = np.linspace(0., 1., 10001)
CANDIDATES = tuple((depth, leaf) for depth in (1, 2, 3) for leaf in (10, 20))
BIN_CANDIDATES = (2, 4, 5, 8, 10, 16)
MISALIGNED_BIN_CANDIDATES = (2, 5, 10)


def mean(x):
    return .18 + .55 * x + .23 * np.sin(2. * np.pi * x) + .16 * x**2


def scale(x):
    return np.select([x < .25, x < .5, x < .75], [.1, .2, .4], default=.8)


def risk(x):
    x = np.asarray(x)
    return np.clip((TARGET - mean(x) + scale(x)) / (2. * scale(x)), 0., 1.)


def basis(x):
    return np.column_stack([np.ones(len(x)), x] +
                           [np.maximum(x - knot, 0.) for knot in KNOTS[1:-1]])


def quantile_model(depth, leaf):
    return GradientBoostingRegressor(
        loss="quantile", alpha=ALPHA, n_estimators=200,
        learning_rate=.05, max_depth=depth, min_samples_leaf=leaf,
        random_state=0)


def fit_selected_quantile(train_x, train_y):
    folds = tuple(KFold(n_splits=3, shuffle=True, random_state=1729).split(train_x))
    candidates = []
    for depth, leaf in CANDIDATES:
        loss = np.mean([
            mean_pinball_loss(
                train_y[validation],
                quantile_model(depth, leaf).fit(
                    train_x[training, None], train_y[training]
                ).predict(train_x[validation, None]),
                alpha=ALPHA,
            )
            for training, validation in folds
        ])
        candidates.append((float(loss), depth, leaf))
    cv_loss, depth, leaf = min(candidates)
    model = quantile_model(depth, leaf).fit(train_x[:, None], train_y)
    return model, {"depth": depth, "min_samples_leaf": leaf,
                   "cv_pinball": cv_loss}


class ResidualBinQuantile:
    """Spline conditional mean plus a fitted lower residual quantile per bin."""

    def __init__(self, x, y, bins):
        self.edges = np.linspace(0., 1., bins + 1)
        self.coefficient = np.linalg.lstsq(basis(x), y, rcond=None)[0]
        group = np.searchsorted(self.edges[1:-1], x, side="right")
        residual = y - basis(x) @ self.coefficient
        counts = np.bincount(group, minlength=bins)
        if np.any(counts < 3):
            raise ValueError("every fitted bin needs at least three observations")
        self.offset = np.asarray([
            np.quantile(residual[group == j], ALPHA) for j in range(bins)
        ])

    def predict(self, x):
        x = np.asarray(x).reshape(-1)
        group = np.searchsorted(self.edges[1:-1], x, side="right")
        return basis(x) @ self.coefficient + self.offset[group]


def fit_selected_binned_quantile(train_x, train_y, candidates=BIN_CANDIDATES):
    folds = tuple(KFold(n_splits=3, shuffle=True, random_state=1729).split(train_x))
    scores = []
    for bins in candidates:
        losses = []
        for training, validation in folds:
            try:
                model = ResidualBinQuantile(train_x[training], train_y[training], bins)
            except ValueError:
                break
            losses.append(mean_pinball_loss(
                train_y[validation], model.predict(train_x[validation]), alpha=ALPHA))
        if len(losses) == len(folds):
            scores.append((float(np.mean(losses)), bins))
    cv_loss, bins = min(scores)
    return ResidualBinQuantile(train_x, train_y, bins), {
        "bins": bins, "cv_pinball": cv_loss}


def first_feasible(lower_bound):
    feasible = np.flatnonzero(lower_bound >= TARGET)
    return float(GRID[feasible[0]]) if len(feasible) else None


def one_run(seed, *, save_curves=False):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0., 1., 480)
    y = mean(x) + scale(x) * rng.uniform(-1., 1., 480)
    train_x, cal_x = x[:240], x[240:]
    train_y, cal_y = y[:240], y[240:]

    quantile, selected = fit_selected_quantile(train_x, train_y)
    raw_cal = quantile.predict(cal_x[:, None])
    correction = conformal_order_statistic(
        raw_cal - cal_y, alpha=ALPHA, level_adjustment=1. / len(cal_x))
    raw_grid = quantile.predict(GRID[:, None])
    direct_x = first_feasible(raw_grid - correction)
    raw_x = first_feasible(raw_grid)

    binned, binned_selected = fit_selected_binned_quantile(train_x, train_y)
    binned_cal = binned.predict(cal_x)
    binned_correction = conformal_order_statistic(
        binned_cal - cal_y, alpha=ALPHA, level_adjustment=1. / len(cal_x))
    binned_grid = binned.predict(GRID)
    binned_x = first_feasible(binned_grid - binned_correction)
    raw_binned_x = first_feasible(binned_grid)

    misaligned, misaligned_selected = fit_selected_binned_quantile(
        train_x, train_y, MISALIGNED_BIN_CANDIDATES)
    misaligned_correction = conformal_order_statistic(
        misaligned.predict(cal_x) - cal_y,
        alpha=ALPHA, level_adjustment=1. / len(cal_x))
    misaligned_x = first_feasible(
        misaligned.predict(GRID) - misaligned_correction)

    # Identical RCC baseline and data split to the earlier picture experiment.
    coefficient = np.linalg.lstsq(basis(train_x), train_y, rcond=None)[0]
    fitted_knots = basis(KNOTS) @ coefficient
    point_model = QuantileSpline(KNOTS, TARGET - fitted_knots,
                                 TARGET - fitted_knots, np.zeros(2))
    rcc_x, _ = solve_learned_rcc(
        point_model, cal_x, TARGET - cal_y,
        alpha=ALPHA, delta=.1, max_splits=64)

    result = {"seed": seed, "selected_model": selected,
              "selected_binned_model": binned_selected,
              "selected_misaligned_model": misaligned_selected,
              "calibration_correction": float(correction),
              "binned_calibration_correction": float(binned_correction),
              "misaligned_calibration_correction": float(misaligned_correction),
              "decisions": {"direct_quantile": direct_x,
                            "raw_quantile": raw_x,
                            "binned_quantile": binned_x,
                            "raw_binned_quantile": raw_binned_x,
                            "misaligned_binned_quantile": misaligned_x,
                            "rcc": rcc_x},
              "risks": {name: None if decision is None else float(risk([decision])[0])
                        for name, decision in (("direct_quantile", direct_x),
                                               ("raw_quantile", raw_x),
                                               ("binned_quantile", binned_x),
                                               ("raw_binned_quantile", raw_binned_x),
                                               ("misaligned_binned_quantile", misaligned_x),
                                               ("rcc", rcc_x))}}
    if save_curves:
        result["curves"] = {
            "x": GRID[::10].tolist(),
            "fitted_lower_quantile": raw_grid[::10].tolist(),
            "calibrated_lower_bound": (raw_grid[::10] - correction).tolist(),
            "binned_lower_quantile": binned_grid[::10].tolist(),
            "calibrated_binned_lower_bound": (
                binned_grid[::10] - binned_correction).tolist(),
            "true_lower_quantile": (mean(GRID[::10]) - .8 * scale(GRID[::10])).tolist(),
        }
    return result


def summarize(runs):
    for name in ("raw_quantile", "direct_quantile", "raw_binned_quantile",
                 "binned_quantile", "misaligned_binned_quantile", "rcc"):
        chosen = [run for run in runs if run["decisions"][name] is not None]
        decisions = np.asarray([run["decisions"][name] for run in chosen])
        risks = np.asarray([run["risks"][name] for run in chosen])
        print(f"{name}: solutions={len(chosen)}/{len(runs)}, "
              f"mean x={np.mean(decisions):.4f}, mean risk={np.mean(risks):.4f}, "
              f"risk <= 10%={np.count_nonzero(risks <= ALPHA + 1e-9)}/{len(chosen)}")
    first = runs[0]
    print("first sample:", first["selected_model"], first["selected_binned_model"],
          first["selected_misaligned_model"],
          "corrections=", first["calibration_correction"],
          first["binned_calibration_correction"],
          "decisions=", first["decisions"], "risks=", first["risks"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent /
                        "results" / "direct-quantile.json")
    args = parser.parse_args()
    if args.replications < 1:
        parser.error("replications must be positive")
    runs = [one_run(20261026 + i, save_curves=(i == 0))
            for i in range(args.replications)]
    summarize(runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"description": __doc__, "runs": runs}, indent=2) + "\n")


if __name__ == "__main__":
    main()
