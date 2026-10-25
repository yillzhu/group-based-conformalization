"""Diagnostics for the scientific interpretation of the enrollment example.

These helpers deliberately distinguish properties of a fitted score from claims
about an unobserved response probability. They do not require Gurobi.
"""

from __future__ import annotations

import numpy as np


def binary_probability_metrics(y_true, probabilities):
    """Return discrimination, loss, and calibration summaries."""

    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("y_true and probabilities must be nonempty and aligned")
    if not np.all(np.isin(y, (0, 1))):
        raise ValueError("y_true must be binary")
    if np.any(~np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probabilities must be finite values in [0, 1]")

    clipped = np.clip(p, 1e-15, 1.0 - 1e-15)
    brier = float(np.mean((p - y) ** 2))
    log_loss = float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)))

    positives = p[y == 1]
    negatives = p[y == 0]
    if positives.size and negatives.size:
        comparisons = positives[:, None] - negatives[None, :]
        auc = float(np.mean(comparisons > 0) + 0.5 * np.mean(comparisons == 0))
    else:
        auc = None

    table = reliability_table(y, p)
    ece = float(
        sum(
            row["fraction"]
            * abs(row["mean_probability"] - row["event_rate"])
            for row in table
        )
    )
    return {
        "n": int(y.size),
        "event_rate": float(np.mean(y)),
        "mean_probability": float(np.mean(p)),
        "brier_score": brier,
        "log_loss": log_loss,
        "roc_auc": auc,
        "ece_10_equal_width": ece,
        "reliability_bins": table,
    }


def reliability_table(y_true, probabilities, n_bins=10):
    """Return nonempty equal-width reliability bins as JSON-safe records."""

    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("y_true and probabilities must be nonempty and aligned")

    bin_index = np.minimum((p * n_bins).astype(int), n_bins - 1)
    result = []
    for index in range(n_bins):
        selected = bin_index == index
        count = int(np.sum(selected))
        if count:
            result.append(
                {
                    "lower": index / n_bins,
                    "upper": (index + 1) / n_bins,
                    "count": count,
                    "fraction": count / y.size,
                    "mean_probability": float(np.mean(p[selected])),
                    "event_rate": float(np.mean(y[selected])),
                }
            )
    return result


def monotonicity_summary(values, *, tolerance=1e-6):
    """Summarize row-wise monotonicity over an increasing decision grid."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError("values must be a 2D array with at least two columns")
    differences = np.diff(array, axis=1)
    violating = np.any(differences < -tolerance, axis=1)
    return {
        "rows": int(array.shape[0]),
        "violating_rows": int(np.sum(violating)),
        "violation_rate": float(np.mean(violating)),
        "minimum_step": float(np.min(differences)),
        "tolerance": float(tolerance),
    }


def paired_decision_summary(candidate, reference, *, equality_tolerance=1e-8):
    """Compare aligned decisions, with positive differences meaning costlier."""

    x = np.asarray(candidate, dtype=float).reshape(-1)
    y = np.asarray(reference, dtype=float).reshape(-1)
    if x.shape != y.shape or x.size == 0:
        raise ValueError("candidate and reference must be nonempty and aligned")
    difference = x - y
    return {
        "n": int(x.size),
        "mean_difference": float(np.mean(difference)),
        "mean_absolute_difference": float(np.mean(np.abs(difference))),
        "median_difference": float(np.median(difference)),
        "minimum_difference": float(np.min(difference)),
        "maximum_difference": float(np.max(difference)),
        "candidate_cheaper": int(np.sum(difference < -equality_tolerance)),
        "equal": int(np.sum(np.abs(difference) <= equality_tolerance)),
        "candidate_costlier": int(np.sum(difference > equality_tolerance)),
    }
