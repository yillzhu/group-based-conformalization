"""Adaptive proposal search with independent, pointwise simulation audits.

The search model has no validity role. At each round the proposal is fixed
before its fresh audit batch is drawn. A union bound across rounds gives a
simultaneous guarantee for every accepted proposal, including a data-adaptive
final choice. It does not certify untested decisions or global optimality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import beta


@dataclass(frozen=True)
class SearchConfig:
    alpha: float = 0.1
    delta: float = 0.05
    rounds: int = 5
    batch_size: int = 500
    proposal_quantile: float = 0.95
    min_separation: float = 0.0
    capacity_index: int | None = None

    def __post_init__(self):
        if not 0 < self.alpha < 1 or not 0 < self.delta < 1:
            raise ValueError("alpha and delta must lie in (0, 1)")
        if self.rounds < 1 or self.batch_size < 1:
            raise ValueError("rounds and batch_size must be positive")
        if not 0 < self.proposal_quantile < 1 or self.min_separation < 0:
            raise ValueError("invalid proposal settings")
        if self.capacity_index is not None and (
            isinstance(self.capacity_index, bool)
            or not isinstance(self.capacity_index, (int, np.integer))
        ):
            raise ValueError("capacity_index must be an integer coordinate")


def binomial_upper_bound(violations: int, size: int, error_probability: float) -> float:
    """One-sided Clopper--Pearson upper bound for a Bernoulli risk."""

    if not 0 <= violations <= size or size < 1 or not 0 < error_probability < 1:
        raise ValueError("invalid binomial audit")
    if violations == size:
        return 1.0
    return float(beta.ppf(1.0 - error_probability, violations + 1, size - violations))


def run_certified_search(
    *,
    candidates: np.ndarray,
    costs: np.ndarray,
    initial_x: np.ndarray,
    initial_response: np.ndarray,
    simulator: Callable[[np.ndarray, int, np.random.Generator], np.ndarray],
    rng: np.random.Generator,
    config: SearchConfig = SearchConfig(),
) -> dict:
    """Search a candidate mesh and audit every proposed point independently.

    ``simulator(x, n, rng)`` returns n iid scalar constraint responses at a
    fixed x; response > 0 denotes violation. It must use only its supplied
    fresh RNG. Previous audits may enter subsequent model fits, but neither
    the current audit data nor its RNG may enter the current proposal.
    """

    from sklearn.ensemble import GradientBoostingRegressor

    candidates = np.asarray(candidates, dtype=float)
    costs = np.asarray(costs, dtype=float).reshape(-1)
    initial_x = np.asarray(initial_x, dtype=float)
    initial_response = np.asarray(initial_response, dtype=float).reshape(-1)
    if candidates.ndim != 2 or candidates.shape[0] == 0:
        raise ValueError("candidates must have shape (n, dimension)")
    if initial_x.ndim != 2 or initial_x.shape[1] != candidates.shape[1]:
        raise ValueError("initial_x has incompatible shape")
    if len(initial_x) != len(initial_response) or len(initial_x) < 2:
        raise ValueError("initial observations are missing or misaligned")
    if len(costs) != len(candidates) or not np.isfinite(costs).all():
        raise ValueError("costs are missing or nonfinite")
    if not np.isfinite(candidates).all() or not np.isfinite(initial_x).all():
        raise ValueError("decisions must be finite")
    if not np.isfinite(initial_response).all():
        raise ValueError("responses must be finite")
    if config.capacity_index is not None and not 0 <= config.capacity_index < candidates.shape[1]:
        raise ValueError("capacity_index must name a decision coordinate")
    if config.capacity_index is not None and candidates.shape[1] < 2:
        raise ValueError("known capacity shift requires another decision coordinate")

    search_x, search_y = initial_x.copy(), initial_response.copy()
    audits = []
    selected = None
    tested = []
    for round_index in range(config.rounds):
        if config.capacity_index is None:
            model_x, model_y, candidate_x = search_x, search_y, candidates
        else:
            # When g(x, scenario) = R(x_without_capacity, scenario) - capacity,
            # learn the random requirement R and retain the exact capacity
            # coefficient. This changes only proposal quality, not the audit.
            index = config.capacity_index
            model_x = np.delete(search_x, index, axis=1)
            model_y = search_y + search_x[:, index]
            candidate_x = np.delete(candidates, index, axis=1)
        model = GradientBoostingRegressor(
            loss="quantile", alpha=config.proposal_quantile,
            n_estimators=120, learning_rate=0.06, max_depth=2,
            min_samples_leaf=20, random_state=round_index,
        ).fit(model_x, model_y)
        predicted_quantile = model.predict(candidate_x)
        if config.capacity_index is not None:
            predicted_quantile -= candidates[:, config.capacity_index]
        available = np.ones(len(candidates), dtype=bool)
        for previous in tested:
            distance = np.linalg.norm(candidates - previous, axis=1)
            available &= distance > config.min_separation + 1e-12
        if not np.any(available):
            break
        predicted_feasible = available & (predicted_quantile <= 0)
        if np.any(predicted_feasible):
            index = min(np.flatnonzero(predicted_feasible), key=lambda i: (costs[i], i))
        else:
            # An exploratory audit can still teach the search model. It does
            # not constitute a feasible or certified decision by itself.
            index = min(np.flatnonzero(available), key=lambda i: (predicted_quantile[i], costs[i], i))
        proposal = candidates[index].copy()
        tested.append(proposal)
        response = np.asarray(simulator(proposal, config.batch_size, rng), dtype=float)
        if response.shape != (config.batch_size,) or not np.isfinite(response).all():
            raise ValueError("simulator must return a finite length-batch vector")
        violations = int(np.count_nonzero(response > 0))
        upper = binomial_upper_bound(
            violations, config.batch_size, config.delta / config.rounds
        )
        accepted = upper <= config.alpha
        audit = {
            "round": round_index + 1, "decision": proposal.tolist(),
            "cost": float(costs[index]), "predicted_quantile": float(predicted_quantile[index]),
            "violations": violations, "batch_size": config.batch_size,
            "empirical_risk": violations / config.batch_size,
            "risk_upper_bound": upper, "accepted": bool(accepted),
        }
        audits.append(audit)
        if accepted and (selected is None or audit["cost"] < selected["cost"]):
            selected = audit
        search_x = np.vstack((search_x, np.repeat(proposal[None, :], config.batch_size, axis=0)))
        search_y = np.concatenate((search_y, response))
    return {
        "selected": selected, "audits": audits,
        "initial_samples": len(initial_x), "audit_samples": config.batch_size * len(audits),
        "alpha": config.alpha, "delta": config.delta,
        "per_round_error_probability": config.delta / config.rounds,
    }
