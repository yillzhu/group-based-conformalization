#!/usr/bin/env python3
"""Plot the first-sample direct-quantile fits from direct_quantile.py."""

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
    first = json.loads((directory / "results" / "direct-quantile.json").read_text())["runs"][0]
    curves = first["curves"]
    x = np.asarray(curves["x"])
    fig, (ax, risk_ax) = plt.subplots(2, 1, figsize=(9.3, 6.2), sharex=True,
                                      gridspec_kw={"height_ratios": [1.2, 1.]})
    ax.plot(x, curves["true_lower_quantile"], color="#245f99", lw=2,
            label="True lower tenth percentile")
    ax.plot(x, curves["calibrated_lower_bound"], color="#b03939", lw=1.6,
            label="Calibrated boosted-tree quantile")
    ax.plot(x, curves["calibrated_binned_lower_bound"], color="#178c91", lw=1.6,
            label="Calibrated binned quantile")
    ax.axhline(.35, color="0.25", ls="--", lw=1, label="Required response 0.35")
    ax.set_ylabel("Lower response bound")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    risk_ax.plot(x, failure_risk(x, .35), color="#245f99", lw=2)
    risk_ax.axhline(.1, color="0.25", ls="--", lw=1)
    risk_ax.set(xlabel="Decision x", ylabel="True P(Y < 0.35 | x)",
                ylim=(-.02, 1.02), xlim=(0., 1.))
    choices = {"Oracle": oracle_decision(.35, .1),
               "Boosted quantile": first["decisions"]["direct_quantile"],
               "Binned quantile": first["decisions"]["binned_quantile"],
               "RCC": first["decisions"]["rcc"]}
    colors = {"Oracle": "#16814a", "Boosted quantile": "#b03939",
              "Binned quantile": "#178c91", "RCC": "#8a5b10"}
    for name, decision in choices.items():
        if decision is None:
            continue
        risk = float(failure_risk([decision], .35)[0])
        risk_ax.scatter(decision, risk, s=70, color=colors[name], zorder=4,
                        label=f"{name}: x={decision:.3f}, risk={risk:.1%}")
    risk_ax.legend(loc="upper right", fontsize=8, frameon=False)
    for axis in (ax, risk_ax):
        for boundary in (.25, .5, .75):
            axis.axvline(boundary, color="0.8", lw=.8, ls=":", zorder=0)
    fig.suptitle("Learning the lower tenth percentile directly: first sample")
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(directory / "figures" / f"direct-quantile.{extension}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
