"""Two-class workforce planning with dedicated and cross-trained capacity."""

from __future__ import annotations

import numpy as np


def staffing_grid(
    dedicated=range(4, 16), flexible=range(0, 19)
) -> np.ndarray:
    """Return decisions ``(d, f)`` with ``d`` dedicated workers per class."""

    return np.asarray([(d, f) for d in dedicated for f in flexible], dtype=float)


def staffing_cost(decisions: np.ndarray, flexible_premium: float = 2.4) -> np.ndarray:
    """Cost two dedicated pools and one cross-trained pool."""

    decisions = _decisions(decisions)
    return 2.0 * decisions[:, 0] + flexible_premium * decisions[:, 1]


def sample_responses(decisions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate unmet workload after optimal use of the flexible pool.

    A positive response means that the staffing decision cannot cover the day's
    workload.  Total workload is lognormal with occasional surges.  A logistic-
    normal class mix creates imbalance between the two dedicated pools.
    """

    decisions = _decisions(decisions)
    n = len(decisions)
    total = rng.lognormal(np.log(180.0) - 0.5 * 0.28**2, 0.28, n)
    total *= 1.0 + 0.35 * (rng.random(n) < 0.08)
    mix = 1.0 / (1.0 + np.exp(-rng.normal(0.0, 1.05, n)))

    dedicated_capacity = 10.0 * decisions[:, 0]
    overflow = np.maximum(0.0, mix * total - dedicated_capacity)
    overflow += np.maximum(0.0, (1.0 - mix) * total - dedicated_capacity)
    return overflow - 9.0 * decisions[:, 1]


def sample_dataset(
    candidates: np.ndarray, size: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sample design points uniformly from the staffing grid and simulate them."""

    if size <= 0:
        raise ValueError("size must be positive")
    candidates = _decisions(candidates)
    x = candidates[rng.integers(0, len(candidates), size=size)]
    return x, sample_responses(x, rng)


def violation_probabilities(
    decisions: np.ndarray, simulations: int, rng: np.random.Generator
) -> np.ndarray:
    """Estimate ``P(response > 0 | decision)`` using common random numbers."""

    decisions = _decisions(decisions)
    if simulations <= 0:
        raise ValueError("simulations must be positive")
    # Resetting scenario draws for each decision would add avoidable noise to
    # comparisons.  Generate one scenario panel and broadcast it instead.
    total = rng.lognormal(
        np.log(180.0) - 0.5 * 0.28**2, 0.28, simulations
    )
    total *= 1.0 + 0.35 * (rng.random(simulations) < 0.08)
    mix = 1.0 / (1.0 + np.exp(-rng.normal(0.0, 1.05, simulations)))
    d = decisions[:, 0, None]
    f = decisions[:, 1, None]
    overflow = np.maximum(0.0, mix[None, :] * total[None, :] - 10.0 * d)
    overflow += np.maximum(
        0.0, (1.0 - mix[None, :]) * total[None, :] - 10.0 * d
    )
    response = overflow - 9.0 * f
    return np.mean(response > 0.0, axis=1)


def _decisions(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("decisions must have shape (n, 2)")
    if not np.isfinite(array).all():
        raise ValueError("decisions must be finite")
    return array
