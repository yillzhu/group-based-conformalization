"""Section 4.6.2: classifier/logit/quantile architectures and interval loss.

Historical implementation; known issues are deliberately retained.
See docs/findings.md and KNOWN-ISSUE markers.
"""

import numpy as np
import torch
import torch.nn as nn


def create_enrollment_net(input_size=3, hidden_size=10, output_size=1):
    """
    Create a one layer ReLU network with 10 hidden neurons for binary classification.

    Architecture: Linear(3 -> 10) -> ReLU -> Linear(10 -> 1) -> Sigmoid

    Note: The sigmoid is included for training to output probabilities,
    but for Gurobi embedding, use create_enrollment_net_for_gurobi() which excludes sigmoid.

    Returns:
        nn.Sequential model
    """
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
        nn.Sigmoid(),  # Output probability in [0, 1]
    )


def create_enrollment_net_for_gurobi(input_size=3, hidden_size=10, output_size=1):
    """
    Create the same network as create_enrollment_net but WITHOUT sigmoid for Gurobi embedding.

    Architecture: Linear(3 -> 10) -> ReLU -> Linear(10 -> 1)

    ✅ Gurobi-compatible: Uses nn.Sequential with only Linear and ReLU layers.
    The output is the logit (pre-sigmoid value). Use inverse_sigmoid(theta) for constraints.

    Returns:
        nn.Sequential model (without sigmoid)
    """
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )


def inverse_sigmoid(p):
    """
    Compute the inverse sigmoid (logit) function: logit(p) = log(p / (1 - p))

    Args:
        p: Probability value(s) in (0, 1)

    Returns:
        Logit value(s)
    """
    return np.log(p / (1.0 - p))


def create_quantile_net(input_size=3, hidden_size=10, output_size=2):
    """
    Create a two hidden layer ReLU network with 10 neurons each for quantile regression.

    Architecture: Linear(3 -> 10) -> ReLU -> Linear(10 -> 10) -> ReLU -> Linear(10 -> 2)

    ✅ Gurobi-compatible: Uses nn.Sequential with only Linear and ReLU layers,
    which are supported by gurobi-machinelearning for embedding into optimization models.

    Returns:
        nn.Sequential model
    """
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )


def interval_score_loss(y_pred, y_true, alpha=0.1):
    """
    Interval score loss for quantile regression.

    ⚠️ POTENTIAL MISTAKE: The reference implementation has the arguments in the
    order (y_true, y_pred, alpha), but PyTorch loss functions typically expect
    (y_pred, y_true). I'm using (y_pred, y_true, alpha) to match PyTorch convention.

    Args:
        y_pred: Predicted quantiles [batch_size, 2] where [:, 0] is lower, [:, 1] is upper
        y_true: True values [batch_size, 1]
        alpha: Miscoverage level (default 0.1 for 90% prediction interval)

    Returns:
        Mean interval score
    """
    ql = y_pred[:, 0:1]  # Lower quantile
    qu = y_pred[:, 1:2]  # Upper quantile

    interval_width = qu - ql

    lower_penalty = (ql - y_true) * (y_true < ql).float()
    upper_penalty = (y_true - qu) * (y_true > qu).float()

    interval_score = interval_width + (2 / alpha) * (lower_penalty + upper_penalty)

    return interval_score.mean()
