"""Sections 4.2 and 4.5.1: global quadratic conformal correction.

Historical implementation; known issues are deliberately retained.
See docs/findings.md and KNOWN-ISSUE markers.
"""

import numpy as np


def conformalizing(quantile, cal_X, cal_y, alpha):
    # Conformalizing
    # Get scores
    q = quantile.predict(cal_X, verbose=0)
    cal_scores = np.maximum(cal_y - q[:, 1:2], q[:, 0:1] - cal_y)
    # Get the score quantile
    cal_n = np.size(cal_y)
    # KNOWN-ISSUE[ISSUE-07]: legacy rank-to-quantile conversion is retained.
    qhat = np.quantile(
        cal_scores, np.ceil((cal_n + 1) * (1 - alpha)) / cal_n, method="higher"
    )
    return qhat
