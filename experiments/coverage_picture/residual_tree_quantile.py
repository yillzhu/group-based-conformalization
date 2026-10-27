#!/usr/bin/env python3
"""Fit a smooth mean and learn residual-heterogeneity regions with one tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

from conformal_cl.order_stats import conformal_order_statistic
from conformal_cl.transparent.learned import solve_learned_rcc
from conformal_cl.transparent.quantile_models import QuantileSpline

from direct_quantile import ALPHA, GRID, KNOTS, TARGET, basis, mean, risk, scale


LEAF_CANDIDATES = (1, 2, 3, 4, 6, 8)
MIN_SAMPLES_CANDIDATES = (10, 20)


class ResidualTreeQuantile:
    """Least-squares spline mean plus an empirical residual quantile per leaf.

    The tree splits to reduce squared-residual heterogeneity. Its leaf
    corrections target the lower tenth percentile of the signed residual.
    """

    def __init__(self, x, y, *, max_leaves, min_samples_leaf):
        self.coefficient = np.linalg.lstsq(basis(x), y, rcond=None)[0]
        residual = y - basis(x) @ self.coefficient
        self.tree = None
        if max_leaves > 1:
            self.tree = DecisionTreeRegressor(
                max_leaf_nodes=max_leaves, min_samples_leaf=min_samples_leaf,
                random_state=0).fit(x[:, None], residual**2)
            leaf = self.tree.apply(x[:, None])
            self.thresholds = np.sort(
                self.tree.tree_.threshold[self.tree.tree_.children_left != -1])
        else:
            leaf = np.zeros(len(x), dtype=int)
            self.thresholds = np.empty(0)
        self.offset = {int(node): float(np.quantile(residual[leaf == node], ALPHA))
                       for node in np.unique(leaf)}
        self.leaf_count = {int(node): int(np.count_nonzero(leaf == node))
                           for node in np.unique(leaf)}

    def predict(self, x):
        x = np.asarray(x).reshape(-1)
        leaf = (self.tree.apply(x[:, None]) if self.tree is not None
                else np.zeros(len(x), dtype=int))
        return basis(x) @ self.coefficient + np.asarray(
            [self.offset[int(node)] for node in leaf])


def fit_selected_tree(x, y):
    folds = tuple(KFold(n_splits=3, shuffle=True, random_state=1729).split(x))
    candidates = []
    for leaves in LEAF_CANDIDATES:
        for minimum in MIN_SAMPLES_CANDIDATES:
            losses = []
            for training, validation in folds:
                model = ResidualTreeQuantile(
                    x[training], y[training], max_leaves=leaves,
                    min_samples_leaf=minimum)
                losses.append(mean_pinball_loss(
                    y[validation], model.predict(x[validation]), alpha=ALPHA))
            candidates.append((float(np.mean(losses)), leaves, minimum))
    cv_loss, leaves, minimum = min(candidates)
    model = ResidualTreeQuantile(
        x, y, max_leaves=leaves, min_samples_leaf=minimum)
    return model, {"max_leaves": leaves, "min_samples_leaf": minimum,
                   "cv_pinball": cv_loss, "actual_leaves": len(model.offset),
                   "thresholds": model.thresholds.tolist(),
                   "leaf_counts": list(model.leaf_count.values())}


def first_feasible(bound):
    indices = np.flatnonzero(bound >= TARGET)
    return float(GRID[indices[0]]) if len(indices) else None


def one_run(seed, *, save_curve=False):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0., 1., 480)
    y = mean(x) + scale(x) * rng.uniform(-1., 1., 480)
    train_x, cal_x = x[:240], x[240:]
    train_y, cal_y = y[:240], y[240:]

    tree, details = fit_selected_tree(train_x, train_y)
    correction = conformal_order_statistic(
        tree.predict(cal_x) - cal_y,
        alpha=ALPHA, level_adjustment=1. / len(cal_x))
    fitted = tree.predict(GRID)
    direct_x = first_feasible(fitted - correction)
    raw_x = first_feasible(fitted)

    fitted_knots = basis(KNOTS) @ np.linalg.lstsq(
        basis(train_x), train_y, rcond=None)[0]
    rcc_model = QuantileSpline(KNOTS, TARGET - fitted_knots,
                               TARGET - fitted_knots, np.zeros(2))
    rcc_x, _ = solve_learned_rcc(
        rcc_model, cal_x, TARGET - cal_y,
        alpha=ALPHA, delta=.1, max_splits=64)

    result = {"seed": seed, "model": details, "calibration_correction": correction,
              "decisions": {"residual_tree": direct_x, "raw_residual_tree": raw_x,
                            "rcc": rcc_x},
              "risks": {name: None if value is None else float(risk([value])[0])
                        for name, value in (("residual_tree", direct_x),
                                            ("raw_residual_tree", raw_x),
                                            ("rcc", rcc_x))}}
    if save_curve:
        grid = GRID[::10]
        result["curve"] = {"x": grid.tolist(),
                           "learned_lower_quantile": fitted[::10].tolist(),
                           "calibrated_lower_bound": (
                               fitted[::10] - correction).tolist(),
                           "true_lower_quantile": (
                               mean(grid) - .8 * scale(grid)).tolist()}
    return result


def summarize(runs):
    for name in ("raw_residual_tree", "residual_tree", "rcc"):
        selected = [run for run in runs if run["decisions"][name] is not None]
        decisions = np.asarray([run["decisions"][name] for run in selected])
        risks = np.asarray([run["risks"][name] for run in selected])
        print(f"{name}: solutions={len(selected)}/{len(runs)}, "
              f"mean x={np.mean(decisions):.4f}, mean risk={np.mean(risks):.4f}, "
              f"risk <= 10%={np.count_nonzero(risks <= ALPHA + 1e-9)}/{len(selected)}")
    print("first sample:", runs[0]["model"],
          "correction=", runs[0]["calibration_correction"],
          "decisions=", runs[0]["decisions"], "risks=", runs[0]["risks"])
    for true_edge in (.25, .5, .75):
        matched = sum(np.any(np.abs(np.asarray(run["model"]["thresholds"])
                                    - true_edge) <= .025) for run in runs)
        print(f"tree has a cut within 0.025 of {true_edge}: {matched}/{len(runs)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent /
                        "results" / "residual-tree-quantile.json")
    args = parser.parse_args()
    if args.replications < 1:
        parser.error("replications must be positive")
    runs = [one_run(20261026 + i, save_curve=(i == 0))
            for i in range(args.replications)]
    summarize(runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"description": __doc__, "runs": runs}, indent=2) + "\n")


if __name__ == "__main__":
    main()
