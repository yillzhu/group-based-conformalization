"""Sections 4.4.2-4.4.3: region state, local calibration, and splitting."""

import numpy as np

from .order_stats import InsufficientCalibrationError, conformal_order_statistic


MIN_CALIBRATION_SIZE = 11


class Group:
    def __init__(
        self,
        bounds=None,
        Nhat=0,
        Xhat=False,
        qhat=0,
        min_x=np.nan,
        max_x=np.nan,
        min_v=np.nan,
        max_v=np.nan,
        ul_x=np.nan,
        ul_v=np.nan,
    ):
        """
        Initialize the Group object with the following attributes:
        - lb: Lower bound
        - ub: Upper bound
        """
        # Each group owns its boundary state.  Sharing this dictionary caused the
        # right-child cut to overwrite the left child and its parent.
        self.bounds = dict(bounds or {})  # [a^T | b] s.t. ax=b is the split plane
        # binary: 1 -> ax<=b; -1 -> ax>=b
        self.Nhat = Nhat  # Default is -1
        self.Xhat = Xhat  # Default is -1
        self.qhat = qhat
        self.min_x = min_x
        self.max_x = max_x
        self.min_v = min_v
        self.max_v = max_v
        self.ul_x = ul_x
        self.ul_v = ul_v

    def conformalize(self, cal_X, R, alpha=0.1, delta=0.1):
        cal_X = np.asarray(cal_X, dtype=float)
        if cal_X.ndim != 2:
            raise ValueError("cal_X must be a two-dimensional array")
        n_cal = len(cal_X)
        if n_cal == 0:
            raise ValueError("cal_X must contain at least one calibration row")
        if not np.isfinite(cal_X).all():
            raise ValueError("cal_X must contain only finite values")
        if len(R) != n_cal:
            raise ValueError("cal_X and R must contain the same number of rows")
        if not 0 < delta < 1:
            raise ValueError(f"delta must be in (0, 1), got {delta!r}")

        index = np.ones(n_cal, dtype=bool)
        for coeffs, d in self.bounds.items():
            coeffs = np.asarray(coeffs, dtype=float)
            if coeffs.size != cal_X.shape[1] + 1 or not np.isfinite(coeffs).all():
                raise ValueError("group bounds must be finite feature coefficients plus b")
            if d not in (-1, 1):
                raise ValueError("group bound directions must be -1 or 1")
            norm = coeffs[:-1]
            b = coeffs[-1]
            # ``np.matmul`` can inherit floating-point status flags from native
            # TensorFlow/Gurobi code on macOS and emit spurious divide-by-zero
            # warnings even for bounded multiplication.  The explicit row-wise
            # contraction is small here and lets us validate the actual result.
            projection = np.einsum("i,ji->j", norm, cal_X)
            if not np.isfinite(projection).all():
                raise FloatingPointError("non-finite group-bound projection")
            index &= d * projection <= d * b

        self.Nhat = np.sum(index)

        sufficiently_represented = self.Nhat >= delta * n_cal * (
            1 - np.sqrt(2 * np.log(n_cal) / delta / n_cal)
        )

        self.Xhat = bool(
            self.Nhat >= MIN_CALIBRATION_SIZE and sufficiently_represented
        )
        if self.Xhat:
            try:
                self.qhat = conformal_order_statistic(
                    R[index], alpha=alpha, level_adjustment=1 / n_cal
                )
            except InsufficientCalibrationError:
                # Keep the inherited correction and route this group through the
                # caller's explicit unresolved-group fallback.  The inherited
                # correction carries only its parent/global interpretation; it
                # does not create a new local-coverage guarantee.
                self.Xhat = False

    def split(self, coeff_split, cal_X, R, alpha=0.1, delta=0.1):
        coeff_split = np.asarray(coeff_split, dtype=float)
        cal_X = np.asarray(cal_X, dtype=float)
        if cal_X.ndim != 2:
            raise ValueError("cal_X must be a two-dimensional array")
        if coeff_split.ndim != 1 or coeff_split.size != cal_X.shape[1] + 1:
            raise ValueError("coeff_split must contain one coefficient per feature plus b")
        if not np.isfinite(coeff_split).all():
            raise ValueError("coeff_split must be finite")

        # Children inherit the parent's correction as an explicit fallback, but
        # own independent boundary dictionaries.
        G_l = Group(self.bounds, qhat=self.qhat)
        G_l.bounds[tuple(coeff_split)] = -1
        G_l.conformalize(cal_X, R, alpha=alpha, delta=delta)

        G_r = Group(self.bounds, qhat=self.qhat)
        G_r.bounds[tuple(coeff_split)] = 1
        G_r.conformalize(cal_X, R, alpha=alpha, delta=delta)

        # Extrema depend on both the region and expression-specific qhat signs.
        # Invalidating them is safer than algebraically shifting a cached value.

        return G_l, G_r

    def describe(self):
        print(self.Nhat, self.Xhat, self.max_x, self.min_x, self.ul_x)


def update_groups(groups_queue, minmax_v, cal_X, R):
    """Retired legacy helper.

    Its former implementation referenced attributes that have never belonged to
    :class:`Group`.  Keeping an explicit failure is safer than producing a partial
    queue mutation followed by an unrelated ``AttributeError``.
    """

    raise NotImplementedError("update_groups is obsolete; use Group.split explicitly")
