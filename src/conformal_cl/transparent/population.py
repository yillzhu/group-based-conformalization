"""Exact population diagnostics for piecewise-uniform capacity observations.

These diagnostics use the known simulation law, never calibration or fitting.
Integration is analytic over the continuous domain, not a decision grid.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


class PopulationScoreLaw:
    """Distribution of max(Y-U(X), L(X)-Y) conditional on an interval."""

    def __init__(self, case, model, left, right):
        if not case.edges[0] <= left < right <= case.edges[-1]:
            raise ValueError("interval must be nonempty and inside the domain")
        points = np.unique(np.r_[left, right,
            model.knots[(model.knots > left) & (model.knots < right)],
            [v for v in case.edges if left < v < right]])
        a, b = points[:-1], points[1:]
        self.weights = (b - a) / (right - left)
        self.scale = case.scale((a + b) / 2.)
        self.start = (model.predict(a) - (1. - a[:, None]) + self.scale[:, None]) / (2. * self.scale[:, None])
        end = (model.predict(b) - (1. - b[:, None]) + self.scale[:, None]) / (2. * self.scale[:, None])
        self.slope = end - self.start

    def cdf(self, correction):
        # On each segment use t in [0,1]. Split at both CDF clipping points
        # and at L-q=U+q. Between these points interval probability is affine.
        lo = self.start[:, 0] - correction / (2. * self.scale)
        hi = self.start[:, 1] + correction / (2. * self.scale)
        dl, du = self.slope[:, 0], self.slope[:, 1]
        candidates = [np.zeros(len(lo)), np.ones(len(lo))]
        for intercept, slope, target in ((lo, dl, 0.), (lo, dl, 1.),
                                          (hi, du, 0.), (hi, du, 1.),
                                          (hi - lo, du - dl, 0.)):
            root = np.divide(target - intercept, slope,
                             out=np.zeros_like(intercept), where=slope != 0.)
            candidates.append(np.clip(root, 0., 1.))
        t = np.sort(np.column_stack(candidates), axis=1)
        probability = np.maximum(
            np.clip(hi[:, None] + du[:, None] * t, 0., 1.)
            - np.clip(lo[:, None] + dl[:, None] * t, 0., 1.), 0.)
        integrals = np.sum(np.diff(t) * (probability[:, :-1] + probability[:, 1:]) / 2., axis=1)
        return float(np.dot(self.weights, integrals))

    def quantile(self, alpha=.1):
        if not 0. < alpha < 1.:
            raise ValueError("alpha must lie in (0,1)")
        lower, upper = -1., 1.
        while self.cdf(lower) >= 1. - alpha:
            lower *= 2.
        while self.cdf(upper) < 1. - alpha:
            upper *= 2.
        return float(brentq(lambda q: self.cdf(q) - (1. - alpha), lower, upper,
                            xtol=1e-13))


def correction_decomposition(case, model, global_correction, local_details, *, alpha=.1):
    """Algebraic decomposition at the adaptively selected RCC cell, not a CI."""

    if "interval" not in local_details:
        return None
    global_law = PopulationScoreLaw(case, model, case.edges[0], case.edges[-1])
    local_law = PopulationScoreLaw(case, model, *local_details["interval"])
    global_population, local_population = global_law.quantile(alpha), local_law.quantile(alpha)
    local_correction = local_details["correction"]
    return {
        "global_population_correction": global_population,
        "local_population_correction": local_population,
        "population_local_minus_global": local_population - global_population,
        "empirical_local_minus_global": float(local_correction - global_correction),
        "sampling_and_selection_remainder": float(
            (local_correction - local_population) - (global_correction - global_population)),
        "global_raw_interval_coverage": global_law.cdf(0.),
        "local_raw_interval_coverage": local_law.cdf(0.),
    }
