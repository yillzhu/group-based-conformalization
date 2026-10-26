"""Learn continuous conditional quantiles from input/response observations only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuantileSpline:
    knots: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    mean_coefficient: np.ndarray

    def __post_init__(self):
        if (len(self.knots) < 2 or self.lower.shape != self.knots.shape
                or self.upper.shape != self.knots.shape):
            raise ValueError("quantile knots and endpoints must align")
        if (not np.isfinite(self.knots).all() or np.any(np.diff(self.knots) <= 0.)
                or not np.isfinite(self.lower).all() or not np.isfinite(self.upper).all()
                or np.any(self.lower > self.upper)):
            raise ValueError("quantiles must be finite, ordered, and have increasing knots")

    def predict(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        return np.column_stack((
            np.interp(x, self.knots, self.lower),
            np.interp(x, self.knots, self.upper),
        ))

    def vertices(self, left, right):
        return np.r_[left, self.knots[(self.knots > left) & (self.knots < right)], right]

    def extrema(self, left, right, endpoint):
        x = self.vertices(left, right)
        value = self.predict(x)[:, endpoint]
        low = int(np.flatnonzero(value == value.min())[0])
        high = int(np.flatnonzero(value == value.max())[-1])
        return float(value[low]), float(x[low]), float(value[high]), float(x[high])

    def cheapest_feasible(self, left, right, correction=0.):
        x = self.vertices(left, right)
        value = self.predict(x)[:, 1] + correction
        for index in range(len(x) - 1):
            if value[index] <= 1e-12:
                return float(x[index])
            if value[index + 1] <= 0.:
                return float(x[index] - value[index] * (x[index + 1] - x[index])
                             / (value[index + 1] - value[index]))
        return float(right) if value[-1] <= 1e-12 else None

    def as_workforce_model(self, endpoint):
        """Equivalent hinge spline for parity tests against the existing solver."""

        from conformal_cl.workforce.models import LinearModel

        values = (self.lower, self.upper)[endpoint]
        slopes = np.diff(values) / np.diff(self.knots)
        coefficient = np.r_[values[0] - slopes[0] * self.knots[0],
                            slopes[0], 0., np.diff(slopes)]
        return LinearModel(
            coefficient, np.zeros(2), np.ones(2),
            dedicated_knots=tuple(self.knots[1:-1]),
        )


def pinball(y, prediction, quantile):
    residual = np.asarray(y) - np.asarray(prediction)
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1.) * residual)))


def fit_quantiles(
    train_x, train_y, validation_x, validation_y, *, domain=(0., 4.),
    alpha=.1, leaf_candidates=(1, 2, 3, 4, 6, 8, 12), seed=0,
    min_samples_leaf=None,
):
    """OLS mean plus a validation-selected residual-quantile tree.

    Squared training residuals supply the tree's split objective. Empirical tail
    quantiles within each leaf supply its two offsets. Transitions interpolate
    across the gap between neighboring training inputs, giving globally ordered,
    continuous piecewise-affine quantiles with no supplied regime boundaries.
    The validation criterion is the sum of lower and upper pinball losses.
    An explicit minimum leaf size overrides the historical sample-fraction
    default, allowing model resolution to grow in finite-data stress studies.
    """

    from sklearn.tree import DecisionTreeRegressor

    train_x, train_y = _data(train_x, train_y, domain)
    validation_x, validation_y = _data(validation_x, validation_y, domain)
    if not 0. < alpha < 1.:
        raise ValueError("alpha must lie in (0,1)")
    if min_samples_leaf is not None and (
        not isinstance(min_samples_leaf, (int, np.integer)) or min_samples_leaf < 1
    ):
        raise ValueError("min_samples_leaf must be a positive integer")
    mean_coefficient = np.linalg.lstsq(
        np.column_stack((np.ones(len(train_x)), train_x)), train_y, rcond=None
    )[0]
    residual = train_y - mean_coefficient[0] - mean_coefficient[1] * train_x
    candidates = []
    for leaves in leaf_candidates:
        if leaves == 1:
            cuts = np.empty(0)
        else:
            tree = DecisionTreeRegressor(
                max_leaf_nodes=leaves,
                min_samples_leaf=(max(40, int(np.ceil(.01 * len(train_x))))
                                  if min_samples_leaf is None else min_samples_leaf),
                random_state=seed,
            ).fit(train_x[:, None], residual ** 2)
            cuts = np.sort(tree.tree_.threshold[tree.tree_.children_left != -1])
        membership = np.searchsorted(cuts, train_x, side="left")
        offsets = np.vstack([
            np.quantile(residual[membership == index], [alpha / 2., 1. - alpha / 2.])
            for index in range(len(cuts) + 1)
        ])
        knots = [domain[0]]
        knot_offsets = [offsets[0]]
        for index, cut in enumerate(cuts):
            before = float(train_x[train_x <= cut].max())
            after = float(train_x[train_x > cut].min())
            knots.extend((before, after))
            knot_offsets.extend((offsets[index], offsets[index + 1]))
        knots.append(domain[1])
        knot_offsets.append(offsets[-1])
        knots = np.asarray(knots)
        base = mean_coefficient[0] + mean_coefficient[1] * knots
        values = np.asarray(knot_offsets) + base[:, None]
        model = QuantileSpline(knots, values[:, 0], values[:, 1], mean_coefficient)
        predicted = model.predict(validation_x)
        loss = pinball(validation_y, predicted[:, 0], alpha / 2.) + pinball(
            validation_y, predicted[:, 1], 1. - alpha / 2.
        )
        candidates.append((loss, len(cuts) + 1, model))
    if not candidates:
        raise ValueError("at least one model candidate is required")
    loss, leaves, selected = min(candidates, key=lambda item: (item[0], item[1]))
    return selected, {
        "selected_leaves": leaves,
        "validation_pinball": float(loss),
        "mean_coefficient": selected.mean_coefficient.tolist(),
        "knots": selected.knots.tolist(),
        "lower_at_knots": selected.lower.tolist(),
        "upper_at_knots": selected.upper.tolist(),
        "candidates": [{"leaves": n, "validation_pinball": float(v)} for v, n, _ in candidates],
    }


def _data(x, y, domain):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (x.ndim != 1 or y.shape != x.shape or len(x) < 40
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or np.any(x <= domain[0]) or np.any(x >= domain[1])):
        raise ValueError("finite aligned observations strictly inside the domain are required")
    return x, y
