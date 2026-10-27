#!/usr/bin/env python3
"""Run the Chapter 4 RCC logic on the data behind the coverage picture.

The optimizer chooses the least x satisfying a fitted, calibrated version of
P(Y < target | x) <= alpha.  The 1-D solver is used because a dummy second
coordinate would create arbitrary geometric cuts absent from this example.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from conformal_cl.order_stats import conformal_order_statistic
from conformal_cl.transparent.learned import solve_learned_rcc
from conformal_cl.transparent.quantile_models import QuantileSpline

from make_figures import EDGES, mean, noise_scale


def basis(x, knots):
    x = np.asarray(x)
    return np.column_stack([np.ones(len(x)), x] +
                           [np.maximum(x - knot, 0.) for knot in knots[1:-1]])


def failure_risk(x, target):
    x = np.asarray(x)
    return np.clip((target - mean(x) + noise_scale(x)) /
                   (2. * noise_scale(x)), 0., 1.)


def oracle_decision(target, alpha):
    # Search each smooth quarter separately, preserving jumps in noise scale.
    for left, right in zip(EDGES[:-1], EDGES[1:]):
        points = np.linspace(left, right, 1001, endpoint=False)
        feasible = mean(points) - (1. - 2. * alpha) * noise_scale(
            np.full(len(points), (left + right) / 2.)) >= target
        if feasible.any():
            index = int(np.flatnonzero(feasible)[0])
            if index == 0:
                return float(left)
            a, b = float(points[index - 1]), float(points[index])
            scale = float(noise_scale(np.asarray([(left + right) / 2.]))[0])
            for _ in range(50):
                mid = (a + b) / 2.
                if mean(mid) - (1. - 2. * alpha) * scale >= target:
                    b = mid
                else:
                    a = mid
            return b
    return None


def run(output_dir, *, target=.35, alpha=.1, seed=20261026):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0., 1., 480)
    y = mean(x) + noise_scale(x) * rng.uniform(-1., 1., 480)
    train_x, calibration_x = x[:240], x[240:]
    train_y, calibration_y = y[:240], y[240:]

    # Fixed *model* knots are neither the true noise boundaries nor RCC cuts.
    knots = np.linspace(0., 1., 6)
    coefficient = np.linalg.lstsq(basis(train_x, knots), train_y, rcond=None)[0]
    fitted_at_knots = basis(knots, knots) @ coefficient
    fitted_mean = lambda values: np.interp(values, knots, fitted_at_knots)

    # Equal lower/upper base predictions make the conformal score the absolute
    # residual.  g=target-Y <= 0 encodes the chance constraint Y >= target.
    model = QuantileSpline(knots, target - fitted_at_knots,
                           target - fitted_at_knots, np.zeros(2))
    residuals = np.abs(calibration_y - fitted_mean(calibration_x))
    global_q = conformal_order_statistic(
        residuals, alpha=alpha, level_adjustment=1. / len(calibration_x))
    global_x = model.cheapest_feasible(0., 1., global_q)
    rcc_x, details = solve_learned_rcc(
        model, calibration_x, target - calibration_y,
        alpha=alpha, delta=.1, max_splits=64)
    oracle_x = oracle_decision(target, alpha)
    selected_count = None
    if rcc_x is not None:
        a, b = details["interval"]
        selected_count = int(np.count_nonzero(
            (calibration_x >= a) & (calibration_x <= b)))

    print(f"seed={seed}; training=240; calibration=240; target={target}; alpha={alpha}")
    for name, decision in (("oracle", oracle_x), ("global", global_x),
                           ("RCC", rcc_x)):
        risk = None if decision is None else float(failure_risk([decision], target)[0])
        print(f"{name}: x={decision}, exact conditional failure risk={risk}")
    print(f"global correction={global_q}; RCC details={details}; "
          f"selected-cell calibration count={selected_count}")

    grid = np.linspace(0., 1., 1001)
    fig, (ax, risk_ax) = plt.subplots(2, 1, figsize=(9.2, 6.3), sharex=True,
                                      gridspec_kw={"height_ratios": [1.25, 1.]})
    ax.scatter(x, y, s=8, color="0.72", alpha=.5, label="480 observed responses")
    ax.plot(grid, mean(grid) - (1. - 2. * alpha) * noise_scale(grid),
            color="#1f5f9e", lw=2, label="true 10th percentile of Y")
    ax.plot(grid, fitted_mean(grid) - global_q, color="#b07808", lw=1.8,
            label="globally calibrated lower bound")
    ax.axhline(target, color="0.2", ls="--", lw=1, label=f"required response {target}")
    ax.set_ylabel("Response / lower bound")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    risk_ax.plot(grid, failure_risk(grid, target), color="#1f5f9e", lw=2)
    risk_ax.axhline(alpha, color="0.2", ls="--", lw=1)
    risk_ax.set_ylabel("True P(Y < target | x)")
    risk_ax.set_xlabel("Decision x")
    risk_ax.set_ylim(-.02, 1.02)
    for axis in (ax, risk_ax):
        for boundary in EDGES[1:-1]:
            axis.axvline(boundary, color="0.8", lw=.8, ls=":", zorder=0)
        axis.set_xlim(0., 1.)
    for name, decision, color in (("oracle", oracle_x, "#16814a"),
                                  ("RCC", rcc_x, "#b03939"),
                                  ("global", global_x, "#8a5b10")):
        if decision is None:
            continue
        for axis in (ax, risk_ax):
            axis.axvline(decision, color=color, lw=1.4, alpha=.9)
        risk_ax.plot(decision, failure_risk([decision], target)[0], "o",
                     color=color, ms=6, label=f"{name}: x={decision:.3f}")
    risk_ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("RCC on the coverage-picture data: minimal x for a 10% failure target")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"rcc-on-picture.{extension}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "figures")
    args = parser.parse_args()
    run(args.output_dir)
