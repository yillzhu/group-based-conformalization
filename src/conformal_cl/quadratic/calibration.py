"""Sections 4.2 and 4.5.1: global quadratic conformal correction."""

import numpy as np

from conformal_cl.order_stats import conformal_order_statistic


def conformalizing(quantile, cal_X, cal_y, alpha):
    # Conformalizing
    # Get scores
    q = quantile.predict(cal_X, verbose=0)
    cal_scores = np.maximum(cal_y - q[:, 1:2], q[:, 0:1] - cal_y)
    return conformal_order_statistic(cal_scores, alpha=alpha)
