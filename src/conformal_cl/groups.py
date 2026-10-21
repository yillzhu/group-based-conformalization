"""Sections 4.4.2-4.4.3: region state, local calibration, and splitting.

Historical implementation; known issues are deliberately retained.
See docs/findings.md and KNOWN-ISSUE markers.
"""

import numpy as np


class Group:
    def __init__(
        self,
        # KNOWN-ISSUE[ISSUE-01]: mutable default and shared child bounds retained.
        bounds={},
        Nhat=-1,
        Xhat=-1,
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
        self.bounds = bounds  # [a^T | b] s.t. ax=b is the split plane
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
        n_cal = len(cal_X)

        index = np.ones(n_cal, dtype=bool)
        for coeffs, d in self.bounds.items():
            coeffs = np.array(coeffs)
            norm = coeffs[:-1]
            b = coeffs[-1]
            index *= d * (norm @ cal_X.T) <= d * b

        self.Nhat = np.sum(index)

        self.Xhat = self.Nhat >= delta * n_cal * (
            1 - np.sqrt(2 * np.log(n_cal) / delta / n_cal)
        )

        # KNOWN-ISSUE[ISSUE-06, ISSUE-07]: strict count and legacy quantile rank retained.
        if self.Nhat > 10 and self.Xhat == 1:
            self.qhat = np.quantile(
                R[index],
                np.ceil((self.Nhat + 1) * (1 - alpha + 1 / n_cal)) / self.Nhat,
                method="higher",
            )

    def split(self, coeff_split, cal_X, R, alpha=0.1, delta=0.1):
        norm_split = coeff_split[:-1]
        b_split = coeff_split[-1]

        # KNOWN-ISSUE[ISSUE-01]: children intentionally share this dictionary.
        G_l = Group(self.bounds)
        G_l.bounds[tuple(coeff_split)] = -1
        G_l.conformalize(cal_X, R, alpha=alpha, delta=delta)

        if norm_split @ self.max_x >= b_split:
            G_l.max_x = self.max_x
            G_l.max_v = self.max_v - self.qhat + G_l.qhat

        if norm_split @ self.min_x >= b_split:
            G_l.min_x = self.min_x
            G_l.min_v = self.min_v - self.qhat + G_l.qhat

        if norm_split @ self.ul_x >= b_split:
            G_l.ul_x = self.ul_x
            G_l.ul_v = self.ul_v - self.qhat + G_l.qhat

        G_r = Group(self.bounds)
        G_r.bounds[tuple(coeff_split)] = 1
        G_r.conformalize(cal_X, R, alpha=alpha, delta=delta)

        if norm_split @ self.max_x <= b_split:
            G_r.max_x = self.max_x
            # KNOWN-ISSUE[ISSUE-03]: left qhat/right cache sign retained.
            G_r.max_v = self.max_v - self.qhat + G_l.qhat

        if norm_split @ self.min_x <= b_split:
            G_r.min_x = self.min_x
            # KNOWN-ISSUE[ISSUE-03]: left qhat/right cache sign retained.
            G_r.min_v = self.min_v - self.qhat + G_l.qhat

        if norm_split @ self.ul_x <= b_split:
            G_r.ul_x = self.ul_x
            # KNOWN-ISSUE[ISSUE-03]: left qhat/right cache sign retained.
            G_r.ul_v = self.ul_v - self.qhat + G_l.qhat

        return G_l, G_r

    def describe(self):
        print(self.Nhat, self.Xhat, self.max_x, self.min_x, self.ul_x)


# KNOWN-ISSUE[ISSUE-09]: obsolete helper references absent Group attributes.
def update_groups(groups_queue, minmax_v, cal_X, R):
    for i in range(groups_queue.qsize()):
        G = groups_queue.get()
        if G.split == 0:
            groups_queue.put(G)
            continue
        elif G.min_v >= minmax_v:
            G.split = 1
            groups_queue.put(G)
            continue
        else:
            bound_split = (G.min_x + G.max_x) / 2

            G_l = Group(G.lb, bound_split)
            G_r = Group(bound_split, G.ub)

            G_l.conformalize(cal_X, R)
            G_r.conformalize(cal_X, R)

            groups_queue.put(G_l)
            groups_queue.put(G_r)
