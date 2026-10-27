#!/usr/bin/env python3
"""Plot learned residual-tree regions and first-sample decisions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_rcc import failure_risk, oracle_decision


def main():
    directory = Path(__file__).resolve().parent
    first = json.loads((directory / "results" /
                        "residual-tree-quantile.json").read_text())["runs"][0]
    curve = first["curve"]
    x = np.asarray(curve["x"])
    fig, (ax, risk_ax) = plt.subplots(2, 1, figsize=(9.3, 6.2), sharex=True,
                                      gridspec_kw={"height_ratios": [1.2, 1.]})
    ax.plot(x, curve["true_lower_quantile"], color="#245f99", lw=2,
            label="True lower tenth percentile")
    ax.plot(x, curve["calibrated_lower_bound"], color="#b03939", lw=1.8,
            label="Mean + tree residual quantile, calibrated")
    ax.axhline(.35, color="0.25", ls="--", lw=1, label="Required response 0.35")
    ax.set_ylabel("Lower response bound")
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    risk_ax.plot(x, failure_risk(x, .35), color="#245f99", lw=2)
    risk_ax.axhline(.1, color="0.25", ls="--", lw=1)
    risk_ax.set(xlabel="Decision x", ylabel="True P(Y < 0.35 | x)",
                ylim=(-.02, 1.02), xlim=(0., 1.))
    choices = {"Oracle": oracle_decision(.35, .1),
               "Residual tree": first["decisions"]["residual_tree"],
               "RCC": first["decisions"]["rcc"]}
    colors = {"Oracle": "#16814a", "Residual tree": "#b03939",
              "RCC": "#8a5b10"}
    for name, decision in choices.items():
        if decision is None:
            continue
        point_risk = float(failure_risk([decision], .35)[0])
        risk_ax.scatter(decision, point_risk, s=70, color=colors[name], zorder=4,
                        label=f"{name}: x={decision:.3f}, risk={point_risk:.1%}")
    risk_ax.legend(loc="upper right", fontsize=8, frameon=False)
    for axis in (ax, risk_ax):
        for boundary in (.25, .5, .75):
            axis.axvline(boundary, color="0.75", lw=1, ls=":", zorder=0)
        for cut in first["model"]["thresholds"]:
            axis.axvline(cut, color="#b03939", lw=.75, ls="--", alpha=.48, zorder=0)
    fig.suptitle("Learned residual-tree regions: first sample")
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(directory / "figures" / f"residual-tree.{extension}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
