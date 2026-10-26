#!/usr/bin/env python3
"""An analytic coverage example inspired by Angelopoulos--Bates Figure 10.

Only the illustration is reproduced conceptually. These fixed-group population
intervals are not the adaptive RCC optimizer in Chapter 4.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


EDGES = np.array([0., .25, .5, .75, 1.])
SCALES = np.array([.1, .2, .4, .8])
TARGET = .9
METHODS = (
    ("Uncalibrated", np.array([.02, .02, .02, .02])),
    ("Marginal 90%", np.array([.48, .48, .48, .48])),
    ("Two-group 90%", np.array([.16, .16, .64, .64])),
    ("Conditional 90%", .9 * SCALES),
)


def mean(x):
    """Smooth nonlinear regression function; coverage calculations center on it."""

    x = np.asarray(x, dtype=float)
    return .18 + .55 * x + .23 * np.sin(2. * np.pi * x) + .16 * x**2


def segment(x):
    """Right-continuous segment index; the value at x=1 is immaterial."""

    return np.clip(np.searchsorted(EDGES[1:-1], x, side="right"), 0, 3)


def noise_scale(x):
    return SCALES[segment(x)]


def interval_halfwidth(x, widths):
    return np.asarray(widths)[segment(x)]


def exact_pointwise_coverage(widths):
    """P(|sigma(x) Z| <= width(x) | x), Z uniform on [-1,1]."""

    return np.minimum(1., np.asarray(widths) / SCALES)


def exact_group_coverage(widths):
    c = exact_pointwise_coverage(widths)
    return (float(np.mean(c[:2])), float(np.mean(c[2:])))


def check_example():
    expected = (
        ((.2, .1, .05, .025), (.15, .0375), .09375),
        ((1., 1., 1., .6), (1., .8), .9),
        ((1., .8, 1., .8), (.9, .9), .9),
        ((.9, .9, .9, .9), (.9, .9), .9),
    )
    for (_, widths), (pointwise, group, marginal) in zip(METHODS, expected):
        np.testing.assert_allclose(exact_pointwise_coverage(widths), pointwise)
        np.testing.assert_allclose(exact_group_coverage(widths), group)
        np.testing.assert_allclose(np.mean(exact_pointwise_coverage(widths)), marginal)


def _save(fig, stem, output_dir):
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"{stem}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_coverage_comparison(output_dir, *, sample_size=480, seed=20261026):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0., 1., sample_size)
    y = mean(x) + noise_scale(x) * rng.uniform(-1., 1., sample_size)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.2), sharex="col",
                             gridspec_kw={"height_ratios": [2.2, 1.]})
    fig.subplots_adjust(left=.07, right=.99, bottom=.19, top=.81, wspace=.16, hspace=.16)
    fig.suptitle("Marginal, group, and pointwise coverage", fontsize=17, y=.98)
    fig.text(.5, .92, "One response model and the same 480 observations in all four columns",
             ha="center", va="center", fontsize=11)

    y_min = min(float(np.min(y)), min(float(np.min(mean(EDGES) - .9 * SCALES.min())),
                                      float(np.min(mean(EDGES) - .48)))) - .10
    y_max = max(float(np.max(y)), float(np.max(mean(EDGES) + .8))) + .10
    for col, (name, widths) in enumerate(METHODS):
        ax, coverage_ax = axes[:, col]
        ax.set_title(name, fontsize=13, pad=11)
        for j in range(4):
            gx = np.linspace(EDGES[j], EDGES[j + 1], 81)
            center = mean(gx)
            ax.fill_between(gx, center - widths[j], center + widths[j],
                            color="#d6a400", alpha=.22, linewidth=0)
            ax.plot(gx, center - widths[j], color="#b08900", lw=1.15)
            ax.plot(gx, center + widths[j], color="#b08900", lw=1.15)
        curve_x = np.linspace(0., 1., 401)
        ax.plot(curve_x, mean(curve_x), color="#222222", lw=1.7, zorder=3)
        covered = np.abs(y - mean(x)) <= interval_halfwidth(x, widths)
        ax.scatter(x[covered], y[covered], s=11, color="#1f9d59", alpha=.67, zorder=4)
        ax.scatter(x[~covered], y[~covered], s=16, marker="x", linewidths=.9,
                   color="#c83135", alpha=.78, zorder=5)
        ax.set_xlim(0., 1.)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([0., .25, .5, .75, 1.])
        if col == 0:
            ax.set_ylabel("Response $Y$")
        else:
            ax.tick_params(labelleft=False)

        coverages = exact_pointwise_coverage(widths)
        coverage_ax.axhline(TARGET, color="#555555", ls="--", lw=1.1, alpha=.75)
        for j, value in enumerate(coverages):
            coverage_ax.hlines(value, EDGES[j], EDGES[j + 1], color="#245caa", lw=3)
            coverage_ax.text((EDGES[j] + EDGES[j + 1]) / 2.,
                             min(value + .055, 1.055), f"{value:.0%}",
                             ha="center", va="bottom", fontsize=9, color="#183c73")
        for edge in EDGES[1:-1]:
            coverage_ax.axvline(edge, color="#888888" if edge == .5 else "#cccccc",
                                lw=1.15 if edge == .5 else .75,
                                ls="--" if edge == .5 else ":", zorder=0)
            ax.axvline(edge, color="#888888" if edge == .5 else "#cccccc",
                       lw=1.15 if edge == .5 else .75,
                       ls="--" if edge == .5 else ":", zorder=0)
        left, right = exact_group_coverage(widths)
        group_label_height = .45 if col == 0 else .10
        coverage_ax.text(.25, group_label_height, f"Group 1: {left:.0%}",
                         ha="center", fontsize=9)
        coverage_ax.text(.75, group_label_height, f"Group 2: {right:.0%}",
                         ha="center", fontsize=9)
        coverage_ax.set_ylim(0., 1.14)
        coverage_ax.set_yticks([0., .5, .9, 1.])
        coverage_ax.set_yticklabels(["0%", "50%", "90%", "100%"])
        coverage_ax.set_xlabel("Input $x$")
        if col == 0:
            coverage_ax.set_ylabel("Coverage at $x$")
        else:
            coverage_ax.tick_params(labelleft=False)

    fig.legend(handles=[
        Line2D([], [], color="#222222", lw=1.7, label="True mean $f(x)$"),
        Line2D([], [], marker="o", ls="none", color="#1f9d59", markersize=5,
               label="Covered observation"),
        Line2D([], [], marker="x", ls="none", color="#c83135", markersize=6,
               label="Missed observation"),
        Line2D([], [], color="#245caa", lw=3, label="Exact coverage at $x$"),
        Line2D([], [], color="#555555", ls="--", lw=1, label="90% target"),
    ], loc="lower center", bbox_to_anchor=(.5, .035), ncol=5, frameon=False)
    _save(fig, "four-coverage-rules", output_dir)


def plot_two_points(output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    fig.subplots_adjust(left=.13, right=.97, bottom=.13, top=.82, wspace=.12)
    fig.suptitle("The group average can hide an error at a chosen point", fontsize=15, y=.96)
    selections = ((.375, "Group 1, wider-noise quarter"),
                  (.875, "Group 2, wider-noise quarter"))
    styles = (("Global", METHODS[1][1], "#40699b", -.15),
              ("Two-group", METHODS[2][1], "#be8100", -.34),
              ("Conditional", METHODS[3][1], "#2c8953", -.53))
    for ax, (x, label) in zip(axes, selections):
        sigma = float(noise_scale(x))
        height = 1. / (2. * sigma)
        ax.fill_between([-sigma, sigma], 0., height, color="#555555", alpha=.18)
        ax.plot([-sigma, -sigma, sigma, sigma], [0., height, height, 0.],
                color="#444444", lw=1.2)
        ax.axvline(0., color="#777777", lw=.9, ls=":")
        ax.axhline(0., color="#444444", lw=.9)
        ax.text(0., height + .16, f"$Y-f(x)$ uniform on [{-sigma:.1f}, {sigma:.1f}]",
                ha="center", fontsize=10)
        for name, widths, color, line_y in styles:
            width = float(interval_halfwidth(x, widths))
            coverage = min(1., width / sigma)
            ax.hlines(line_y, -width, width, color=color, lw=6)
            ax.plot([-width, width], [line_y, line_y], "|", color=color,
                    markersize=13, markeredgewidth=1.4)
            ax.text(-1.16, line_y, name, va="center", color=color, fontsize=10)
            ax.text(1.16, line_y, f"{coverage:.0%}", ha="right", va="center",
                    color=color, fontsize=10)
        ax.set_xlim(-1.19, 1.19)
        ax.set_ylim(-.65, 2.9)
        ax.set_title(f"$x={x:.3f}$ — {label}", fontsize=11, pad=12)
        ax.set_xlabel("Residual $Y-f(x)$")
        ax.set_yticks([0., 1., 2.])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_ylabel("Conditional density (above zero)")
    _save(fig, "two-selected-points", output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "figures")
    args = parser.parse_args()
    check_example()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_coverage_comparison(args.output_dir)
    plot_two_points(args.output_dir)
    for name, widths in METHODS:
        pointwise = exact_pointwise_coverage(widths)
        g1, g2 = exact_group_coverage(widths)
        print(f"{name:17s} pointwise={pointwise} groups=({g1:.3f}, {g2:.3f}) "
              f"marginal={np.mean(pointwise):.3f}")


if __name__ == "__main__":
    main()
