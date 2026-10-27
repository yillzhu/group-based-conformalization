#!/usr/bin/env python3
"""Compare RCC with simple fixed partitions on the coverage-picture problem."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from conformal_cl.order_stats import conformal_order_statistic
from conformal_cl.transparent.learned import solve_learned_rcc
from conformal_cl.transparent.quantile_models import QuantileSpline

from make_figures import mean, noise_scale
from run_rcc import basis, failure_risk, oracle_decision


TARGET = .35
ALPHA = .1
KNOTS = np.linspace(0., 1., 6)
GRID = np.linspace(0., 1., 10001)


def fixed_partition_decision(x, residual, fitted_mean, edges):
    group = np.searchsorted(edges[1:-1], x, side="right")
    correction = np.asarray([
        conformal_order_statistic(
            residual[group == j], alpha=ALPHA, level_adjustment=1. / len(x))
        if np.count_nonzero(group == j) >= 11 else np.inf
        for j in range(len(edges) - 1)
    ])
    grid_group = np.searchsorted(edges[1:-1], GRID, side="right")
    feasible = fitted_mean(GRID) - correction[grid_group] >= TARGET
    return float(GRID[np.flatnonzero(feasible)[0]]) if feasible.any() else None


def one_run(seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0., 1., 480)
    y = mean(x) + noise_scale(x) * rng.uniform(-1., 1., 480)
    train_x, cal_x = x[:240], x[240:]
    train_y, cal_y = y[:240], y[240:]
    coefficient = np.linalg.lstsq(basis(train_x, KNOTS), train_y, rcond=None)[0]
    fitted_knots = basis(KNOTS, KNOTS) @ coefficient
    fitted_mean = lambda points: np.interp(points, KNOTS, fitted_knots)
    residual = np.abs(cal_y - fitted_mean(cal_x))
    model = QuantileSpline(KNOTS, TARGET - fitted_knots,
                           TARGET - fitted_knots, np.zeros(2))
    global_q = conformal_order_statistic(
        residual, alpha=ALPHA, level_adjustment=1. / len(cal_x))
    rcc, _ = solve_learned_rcc(model, cal_x, TARGET - cal_y,
                               alpha=ALPHA, delta=.1, max_splits=64)
    return {
        "RCC": rcc,
        "Global": model.cheapest_feasible(0., 1., global_q),
        "Two fixed halves": fixed_partition_decision(
            cal_x, residual, fitted_mean, np.asarray([0., .5, 1.])),
        "Ten fixed bins": fixed_partition_decision(
            cal_x, residual, fitted_mean, np.linspace(0., 1., 11)),
    }


def main():
    seeds = range(20261026, 20261126)
    results = [one_run(seed) for seed in seeds]
    print("100 independent samples; 240 training + 240 calibration observations each")
    for name in results[0]:
        decisions = np.asarray([run[name] for run in results if run[name] is not None])
        risks = failure_risk(decisions, TARGET)
        print(f"{name}: solutions={len(decisions)}/100, "
              f"mean x={np.mean(decisions):.4f}, "
              f"mean risk={np.mean(risks):.4f}, "
              f"risk <= 10%={np.count_nonzero(risks <= ALPHA + 1e-9)}/{len(risks)}")

    fig, ax = plt.subplots(figsize=(9.2, 4.5))
    dense = np.linspace(0., 1., 2001)
    ax.plot(dense, failure_risk(dense, TARGET), color="#245f99", lw=2,
            label="True failure risk")
    ax.axhline(ALPHA, color="0.25", ls="--", lw=1, label="10% target")
    choices = {"Oracle": oracle_decision(TARGET, ALPHA), **results[0]}
    colors = {"Oracle": "#16814a", "RCC": "#b03939", "Global": "#8a5b10",
              "Two fixed halves": "#6b4c9a", "Ten fixed bins": "#178c91"}
    for name, decision in choices.items():
        if decision is None:
            continue
        risk = float(failure_risk([decision], TARGET)[0])
        ax.scatter(decision, risk, s=75, color=colors[name], zorder=4,
                   label=f"{name}: x={decision:.3f}, risk={risk:.1%}")
    ax.set(xlim=(0., 1.), ylim=(-.02, 1.02), xlabel="Decision x",
           ylabel="True P(Y < 0.35 | x)",
           title="Same 480 observations: RCC and simpler alternatives")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    output = Path(__file__).resolve().parent / "figures"
    for extension in ("png", "svg"):
        fig.savefig(output / f"baseline-comparison.{extension}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
