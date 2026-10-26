"""One-dimensional capacity choice with an exact mean and piecewise noise scale.

The population experiment isolates partition geometry from prediction and
calibration error. The empirical experiment preserves the original RCC use of
the same calibration sample for splitting and recalibration, including its
unresolved-region fallback. Neither is labeled an optimizer-valid theorem.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from conformal_cl.groups import Group
from conformal_cl.order_stats import (
    InsufficientCalibrationError,
    conformal_order_statistic,
)


@dataclass(frozen=True)
class CapacityCase:
    name: str
    edges: tuple[float, ...]
    scales: tuple[float, ...]

    def __post_init__(self):
        if len(self.edges) != len(self.scales) + 1:
            raise ValueError("one scale is required per interval")
        if not np.isfinite(self.edges).all() or np.any(np.diff(self.edges) <= 0.):
            raise ValueError("edges must be finite and strictly increasing")
        if not np.isfinite(self.scales).all() or min(self.scales) <= 0.:
            raise ValueError("scales must be finite and positive")

    def scale(self, x):
        index = np.searchsorted(self.edges[1:-1], x, side="right")
        return np.asarray(self.scales)[index]

    def risk(self, x):
        """Exact P(1 - x + sigma(x) Z > 0), Z uniform on [-1, 1]."""

        return np.clip((1. - (np.asarray(x) - 1.) / self.scale(x)) / 2., 0., 1.)

    def draw(self, size, rng):
        if size <= 0:
            raise ValueError("size must be positive")
        x = rng.uniform(self.edges[0], self.edges[-1], size=size)
        residual = self.scale(x) * rng.uniform(-1., 1., size=size)
        return x, residual


CASES = (
    CapacityCase("homogeneous", (0., 4.), (.1,)),
    CapacityCase("aligned", (0., 1.2, 4.), (.1, .4)),
    CapacityCase("interior_quiet_region", (0., .9, 1.4, 4.), (.4, .1, .4)),
)


def population_width(case, left, right, alpha=.1):
    """Exact (1-alpha)-quantile of |sigma(X) Z| given left <= X <= right."""

    if not 0. < alpha < 1. or not left < right:
        raise ValueError("invalid quantile or interval")
    overlap = np.maximum(
        0., np.minimum(right, case.edges[1:]) - np.maximum(left, case.edges[:-1])
    )
    if overlap.sum() <= 0.:
        raise ValueError("interval does not intersect the domain")
    weights = overlap / overlap.sum()
    scales = np.asarray(case.scales)
    saturated = 0.
    for scale in np.unique(scales[weights > 0.]):
        remaining = (scales >= scale) & (weights > 0.)
        proposal = (1. - alpha - saturated) / np.sum(weights[remaining] / scales[remaining])
        if proposal <= scale + 1e-14:
            return float(proposal)
        saturated += float(weights[scales == scale].sum())
    raise ArithmeticError("failed to invert the mixture CDF")


def _best_decision(case, widths):
    """Solve min x, 1 - x + width_j <= 0 on each true scale interval."""

    candidates = []
    for left, right, width in zip(case.edges[:-1], case.edges[1:], widths):
        x = max(left, 1. + width)
        # Cells are left closed/right open, except for the last cell.
        if x < right or (right == case.edges[-1] and x <= right):
            candidates.append(x)
    return min(candidates) if candidates else None


def _report(case, decision, **details):
    if decision is None:
        return {"status": "no_solution", **details}
    return {
        "status": "selected", "capacity_cost": float(decision),
        "exact_violation_probability": float(case.risk(decision)), **details,
    }


def solve_rcc(case, *, alpha=.1, delta=.1, max_splits=64, data=None):
    """Chapter-style RCC on continuous intervals, using exact affine extrema.

    Base endpoints both equal the exact mean 1-x. CQR scores are |residual|.
    With data=None, corrections are population quantiles and mass tests are
    exact. Otherwise Group supplies the historical count rule and adjusted
    rank. All undersized/pending cells retain the constrained fallback.
    """

    if not 0. < alpha < 1. or not 0. < delta < 1. or max_splits < 0:
        raise ValueError("invalid RCC settings")
    domain_left, domain_right = case.edges[0], case.edges[-1]
    domain_length = domain_right - domain_left
    if data is not None:
        x, residual = map(np.asarray, data)
        if x.ndim != 1 or residual.shape != x.shape or not len(x):
            raise ValueError("data must contain aligned one-dimensional arrays")
        scores = np.abs(residual)

    def calibrate(left, right, inherited):
        if data is None:
            represented = (right - left) / domain_length >= delta - 1e-14
            width = population_width(case, left, right, alpha) if represented else inherited
        else:
            group = Group(
                bounds={(1., right): 1, (-1., -left): 1}, qhat=inherited
            )
            group.conformalize(x[:, None], scores, alpha=alpha, delta=delta)
            represented, width = group.Xhat, group.qhat
        return (left, right, float(width), bool(represented))

    initial_width = (
        population_width(case, domain_left, domain_right, alpha)
        if data is None
        else conformal_order_statistic(scores, alpha=alpha, level_adjustment=1. / len(x))
    )
    root = calibrate(domain_left, domain_right, initial_width)
    queue = deque([root]) if root[3] else deque()
    unresolved = [] if root[3] else [root]
    safe = []
    discarded = splits = 0
    while queue and splits < max_splits:
        cell = queue.popleft()
        left, right, width, _ = cell
        upper_max, upper_min = 1. - left + width, 1. - right + width
        lower_min = 1. - right - width
        if upper_max <= 1e-12:
            safe.append(cell)
            continue
        if lower_min > 0.:
            discarded += 1
            continue
        split = 1. + width if upper_min <= 0. else (left + right) / 2.
        if split <= left + 1e-12 or split >= right - 1e-12:
            unresolved.append(cell)
            continue
        for a, b in ((left, split), (split, right)):
            child = calibrate(a, b, width)
            (queue if child[3] else unresolved).append(child)
        splits += 1
    unresolved.extend(queue)

    proposals = []
    for route, cells in (("safe", safe), ("unresolved", unresolved)):
        for left, right, width, represented in cells:
            decision = left if route == "safe" else max(left, 1. + width)
            if decision <= right + 1e-12:
                proposals.append((decision, left, right, width, route, represented))
    if not proposals:
        return _report(case, None, splits=splits, discarded_regions=discarded)
    decision, left, right, width, route, represented = min(proposals)
    return _report(
        case, decision, interval=[left, right], correction_width=width,
        route=route, sufficiently_represented=represented, splits=splits,
        safe_regions=len(safe), unresolved_regions=len(unresolved),
        selected_cell_crosses_scale_boundary=any(left < t < right for t in case.edges[1:-1]),
    )


def population_comparison(case, *, alpha=.1, delta=.1):
    """Compare geometry with local calibration and a correctly scaled predictor."""

    global_width = population_width(case, case.edges[0], case.edges[-1], alpha)
    local_widths = (1. - alpha) * np.asarray(case.scales)
    oracle_widths = (1. - 2. * alpha) * np.asarray(case.scales)
    return {
        "case": case.name, "edges": case.edges, "scales": case.scales,
        "methods": {
            "global_cqr": _report(case, _best_decision(case, [global_width] * len(case.scales))),
            "rcc": solve_rcc(case, alpha=alpha, delta=delta),
            "known_regions_cqr": _report(case, _best_decision(case, local_widths)),
            "known_scale_cqr": _report(case, _best_decision(case, local_widths)),
            "chance_oracle": _report(case, _best_decision(case, oracle_widths)),
        },
    }


def tolerance_rank(size, *, alpha, beta):
    """Smallest k with P(Bin(size,1-alpha) >= k) <= beta; None if unavailable."""

    from scipy.stats import binom

    if size <= 0 or not 0. < alpha < 1. or not 0. < beta < 1.:
        raise ValueError("invalid tolerance-rank settings")
    rank = int(binom.ppf(1. - beta, size, 1. - alpha)) + 1
    return rank if rank <= size else None


def empirical_comparison(case, size, rng, *, alpha=.1, beta=.05, delta=.1):
    """Same iid labeled observations for every calibration method; risk is analytic."""

    x, residual = case.draw(size, rng)
    adjustment = 1. / size
    global_width = conformal_order_statistic(
        np.abs(residual), alpha=alpha, level_adjustment=adjustment
    )
    scaled_width = conformal_order_statistic(
        np.abs(residual) / case.scale(x), alpha=alpha, level_adjustment=adjustment
    )
    regions_widths, tolerance_widths, ranks = [], [], []
    for index, (left, right) in enumerate(zip(case.edges[:-1], case.edges[1:])):
        mask = (x >= left) & (x < right)
        scores = residual[mask]
        try:
            width = conformal_order_statistic(
                np.abs(scores), alpha=alpha, level_adjustment=adjustment
            )
        except InsufficientCalibrationError:
            width = np.inf
        regions_widths.append(width)
        rank = tolerance_rank(len(scores), alpha=alpha, beta=beta / len(case.scales)) if len(scores) else None
        tolerance_widths.append(
            float(np.partition(scores, rank - 1)[rank - 1]) if rank is not None else np.inf
        )
        ranks.append({"region": index, "count": len(scores), "rank": rank})
    return {
        "global_cqr": _report(case, _best_decision(case, [global_width] * len(case.scales))),
        "rcc": solve_rcc(case, alpha=alpha, delta=delta, data=(x, residual)),
        "known_regions_cqr": _report(case, _best_decision(case, regions_widths)),
        "known_scale_cqr": _report(case, _best_decision(case, scaled_width * np.asarray(case.scales))),
        "simultaneous_tolerance": _report(case, _best_decision(case, tolerance_widths), calibration_ranks=ranks),
    }
