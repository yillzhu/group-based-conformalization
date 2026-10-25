"""Continuous-capacity call-center model using square-root safety staffing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContinuousQueueConfig:
    offered_load: float = 6.0
    safety_factor: float = 1.6
    daily_volume_sigma: float = 0.18
    surge_probability: float = 0.08
    surge_multiplier: float = 1.35
    class_mix_logit_sigma: float = 0.85
    flexible_efficiency: float = 0.9
    dedicated_cost: float = 2.0
    flexible_cost: float = 1.4


def sample_continuous_queue_responses(
    decisions: np.ndarray,
    rng: np.random.Generator,
    config: ContinuousQueueConfig,
) -> np.ndarray:
    """Return required flexible FTE minus the planned flexible capacity.

    For each class, the square-root staffing approximation requires offered
    load plus ``safety_factor * sqrt(load)`` capacity. Dedicated FTE is used
    first; a shared flexible pool covers the remaining requirements.
    """

    decisions = _decisions(decisions)
    requirements = _sample_requirements(len(decisions), rng, config)
    dedicated = decisions[:, 0, None]
    required_flexible = np.maximum(requirements - dedicated, 0.0).sum(axis=1)
    required_flexible /= config.flexible_efficiency
    return required_flexible - decisions[:, 1]


def sample_continuous_queue_dataset(
    lower: np.ndarray,
    upper: np.ndarray,
    size: int,
    rng: np.random.Generator,
    config: ContinuousQueueConfig,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(lower, dtype=float).reshape(2)
    upper = np.asarray(upper, dtype=float).reshape(2)
    if np.any(lower >= upper) or size <= 0:
        raise ValueError("invalid continuous queue sampling domain")
    decisions = rng.uniform(lower, upper, size=(size, 2))
    return decisions, sample_continuous_queue_responses(decisions, rng, config)


def continuous_queue_violation_probabilities(
    decisions: np.ndarray,
    simulations: int,
    rng: np.random.Generator,
    config: ContinuousQueueConfig,
) -> np.ndarray:
    decisions = _decisions(decisions)
    requirements = _sample_requirements(simulations, rng, config)
    result = np.empty(len(decisions))
    for index, (dedicated, flexible) in enumerate(decisions):
        required_flexible = np.maximum(requirements - dedicated, 0.0).sum(axis=1)
        required_flexible /= config.flexible_efficiency
        result[index] = np.mean(required_flexible > flexible)
    return result


def approximate_continuous_queue_oracle(
    *,
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    alpha: float,
    simulations: int,
    rng: np.random.Generator,
    config: ContinuousQueueConfig,
    dedicated_grid_size: int = 2001,
) -> tuple[np.ndarray, float, float]:
    lower = np.asarray(domain_lower, dtype=float).reshape(2)
    upper = np.asarray(domain_upper, dtype=float).reshape(2)
    if not 0.0 < alpha < 1.0 or simulations <= 0 or dedicated_grid_size < 2:
        raise ValueError("invalid continuous queue oracle parameters")
    requirements = _sample_requirements(simulations, rng, config)
    rank = min(simulations - 1, int(np.ceil((1.0 - alpha) * simulations)) - 1)
    best = None
    for dedicated in np.linspace(lower[0], upper[0], dedicated_grid_size):
        required_flexible = np.maximum(requirements - dedicated, 0.0).sum(axis=1)
        required_flexible /= config.flexible_efficiency
        flexible = max(
            lower[1],
            float(np.partition(required_flexible, rank)[rank]),
        )
        if flexible > upper[1]:
            continue
        cost = config.dedicated_cost * dedicated + config.flexible_cost * flexible
        violation = float(np.mean(required_flexible > flexible))
        proposal = (cost, dedicated, flexible)
        if best is None or proposal < best[0]:
            best = (proposal, violation)
    if best is None:
        raise ValueError("continuous queue oracle found no feasible decision")
    proposal, violation = best
    return np.asarray(proposal[1:]), float(proposal[0]), violation


def _sample_requirements(
    size: int,
    rng: np.random.Generator,
    config: ContinuousQueueConfig,
) -> np.ndarray:
    volume = rng.lognormal(
        -0.5 * config.daily_volume_sigma**2,
        config.daily_volume_sigma,
        size,
    )
    volume *= np.where(
        rng.random(size) < config.surge_probability,
        config.surge_multiplier,
        1.0,
    )
    total_load = config.offered_load * volume
    mix = 1.0 / (
        1.0 + np.exp(-rng.normal(0.0, config.class_mix_logit_sigma, size))
    )
    loads = np.column_stack((mix * total_load, (1.0 - mix) * total_load))
    return loads + config.safety_factor * np.sqrt(loads)


def _decisions(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("decisions must be a finite array with shape (n, 2)")
    return values
