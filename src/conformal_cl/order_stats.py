"""Finite-sample order statistics used by conformal calibration."""

from __future__ import annotations

import math

import numpy as np


class InsufficientCalibrationError(ValueError):
    """Raised when the requested finite-sample rank is unavailable."""


def conformal_order_statistic(
    scores,
    *,
    alpha: float,
    level_adjustment: float = 0.0,
) -> float:
    """Return the exact finite-sample conformal order statistic.

    The selected one-based rank is

    ``ceil((n + 1) * (1 - alpha + level_adjustment))``.

    Selecting the rank directly avoids the off-by-one behavior that results from
    passing ``rank / n`` to :func:`numpy.quantile`.  If the requested rank is
    larger than the available sample, no finite empirical correction exists and
    the caller must apply an explicit fallback policy.
    """

    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    if level_adjustment < 0:
        raise ValueError("level_adjustment must be nonnegative")

    values = np.asarray(scores, dtype=float).reshape(-1)
    if values.size == 0:
        raise InsufficientCalibrationError("calibration scores are empty")
    if not np.isfinite(values).all():
        raise ValueError("calibration scores must all be finite")

    rank = math.ceil((values.size + 1) * (1 - alpha + level_adjustment))
    if rank < 1 or rank > values.size:
        raise InsufficientCalibrationError(
            f"requested rank {rank} is unavailable for {values.size} scores"
        )

    return float(np.partition(values, rank - 1)[rank - 1])
