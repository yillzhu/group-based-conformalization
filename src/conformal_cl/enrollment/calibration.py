"""Section 4.6.2: teacher-probability interval calibration.

Historical implementation; known issues are deliberately retained.
See docs/findings.md and KNOWN-ISSUE markers.
"""

import numpy as np
import torch


def conformalizing(model, cal_X, cal_y, alpha):
    """
    Conformal prediction calibration.

    Reference: quadratic_sample.py conformalizing function

    Args:
        model: Trained quantile model (outputs [lower, upper] quantiles)
        cal_X: Calibration features (tensor)
        cal_y: Calibration labels (numpy array)
        alpha: Miscoverage level

    Returns:
        qhat: Conformal quantile adjustment value
    """
    # Get quantile predictions
    model.eval()
    with torch.no_grad():
        q = model(cal_X).numpy()

    # Calculate nonconformity scores: max(y - upper, lower - y)
    cal_scores = np.maximum(cal_y - q[:, 1:2], q[:, 0:1] - cal_y)

    # Get the score quantile
    cal_n = cal_y.shape[0]
    # KNOWN-ISSUE[ISSUE-07]: legacy rank-to-quantile conversion is retained.
    qhat = np.quantile(
        cal_scores, np.ceil((cal_n + 1) * (1 - alpha)) / cal_n, method="higher"
    )

    return qhat
