"""Section 4.6.2: teacher-probability interval calibration."""

import numpy as np
import torch

from conformal_cl.order_stats import conformal_order_statistic


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

    return conformal_order_statistic(cal_scores, alpha=alpha)
