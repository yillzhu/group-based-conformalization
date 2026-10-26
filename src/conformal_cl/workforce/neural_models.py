"""Optional PyTorch training for small continuous-workforce ReLU models."""

from __future__ import annotations

import numpy as np

from .models import ReLUModel


def fit_relu_mean(
    x: np.ndarray,
    y: np.ndarray,
    *,
    hidden_units: int = 16,
    epochs: int = 750,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-4,
    seed: int = 0,
) -> ReLUModel:
    """Fit a one-hidden-layer point predictor using standardized MSE."""

    _validate_training_options(hidden_units, epochs, learning_rate, weight_decay)
    torch, nn = _torch()
    inputs, response, center, scale, response_center, response_scale = _data(
        x, y, torch
    )
    torch.manual_seed(seed)
    network = nn.Sequential(
        nn.Linear(2, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, 1),
    )
    optimizer = torch.optim.Adam(
        network.parameters(), learning_rate, weight_decay=weight_decay
    )
    for _ in range(epochs):
        optimizer.zero_grad()
        prediction = network(inputs).reshape(-1)
        loss = torch.mean((prediction - response) ** 2)
        loss.backward()
        optimizer.step()
    return _extract(network, 0, center, scale, response_center, response_scale)


def fit_relu_quantiles(
    x: np.ndarray,
    y: np.ndarray,
    lower_quantile: float,
    upper_quantile: float,
    *,
    hidden_units: int = 16,
    epochs: int = 750,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-4,
    seed: int = 0,
) -> tuple[ReLUModel, ReLUModel]:
    """Fit globally noncrossing quantiles with shared ReLU features."""

    if not 0.0 < lower_quantile < upper_quantile < 1.0:
        raise ValueError("quantile levels must be ordered inside (0, 1)")
    _validate_training_options(hidden_units, epochs, learning_rate, weight_decay)
    torch, nn = _torch()
    inputs, response, center, scale, response_center, response_scale = _data(
        x, y, torch
    )
    torch.manual_seed(seed)

    class NoncrossingQuantiles(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(2, hidden_units)
            self.lower = nn.Linear(hidden_units, 1)
            self.width_weight_raw = nn.Parameter(torch.zeros(hidden_units))
            self.width_bias_raw = nn.Parameter(torch.tensor(0.0))

        def forward(self, values):
            hidden = torch.relu(self.hidden(values))
            lower = self.lower(hidden).reshape(-1)
            width = torch.nn.functional.softplus(self.width_bias_raw)
            width = width + hidden @ torch.nn.functional.softplus(
                self.width_weight_raw
            )
            return torch.column_stack((lower, lower + width))

    network = NoncrossingQuantiles()
    optimizer = torch.optim.Adam(
        network.parameters(), learning_rate, weight_decay=weight_decay
    )
    for _ in range(epochs):
        optimizer.zero_grad()
        prediction = network(inputs)
        lower_loss = _pinball(torch, response - prediction[:, 0], lower_quantile)
        upper_loss = _pinball(torch, response - prediction[:, 1], upper_quantile)
        loss = lower_loss + upper_loss
        loss.backward()
        optimizer.step()
    hidden_weight = network.hidden.weight.detach().cpu().numpy().astype(float)
    hidden_bias = network.hidden.bias.detach().cpu().numpy().astype(float)
    lower_weight = network.lower.weight[0].detach().cpu().numpy().astype(float)
    lower_bias = float(network.lower.bias[0].detach().cpu().item())
    width_weight = (
        torch.nn.functional.softplus(network.width_weight_raw)
        .detach()
        .cpu()
        .numpy()
        .astype(float)
    )
    width_bias = float(
        torch.nn.functional.softplus(network.width_bias_raw).detach().cpu().item()
    )
    return (
        _from_parameters(
            hidden_weight,
            hidden_bias,
            lower_weight,
            lower_bias,
            center,
            scale,
            response_center,
            response_scale,
        ),
        _from_parameters(
            hidden_weight,
            hidden_bias,
            lower_weight + width_weight,
            lower_bias + width_bias,
            center,
            scale,
            response_center,
            response_scale,
        ),
    )


def _pinball(torch, residual, quantile):
    return torch.mean(torch.maximum(quantile * residual, (quantile - 1.0) * residual))


def _validate_training_options(hidden_units, epochs, learning_rate, weight_decay):
    if hidden_units <= 0 or epochs <= 0:
        raise ValueError("hidden_units and epochs must be positive")
    if learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("invalid optimizer settings")


def _data(x, y, torch):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.ndim != 2 or x.shape[1] != 2 or len(x) != len(y):
        raise ValueError("x and y must be aligned two-dimensional training data")
    center = (x.min(axis=0) + x.max(axis=0)) / 2.0
    scale = (x.max(axis=0) - x.min(axis=0)) / 2.0
    if np.any(scale <= 0.0) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("training data must be finite with varying features")
    response_center = float(np.mean(y))
    response_scale = float(np.std(y))
    if response_scale <= 0.0:
        raise ValueError("training responses must vary")
    inputs = torch.tensor((x - center) / scale, dtype=torch.float32)
    response = torch.tensor(
        (y - response_center) / response_scale, dtype=torch.float32
    )
    return inputs, response, center, scale, response_center, response_scale


def _extract(network, output_index, center, scale, response_center, response_scale):
    first = network[0]
    last = network[2]
    hidden_weight = first.weight.detach().cpu().numpy().astype(float)
    hidden_bias = first.bias.detach().cpu().numpy().astype(float)
    return _from_parameters(
        hidden_weight,
        hidden_bias,
        last.weight[output_index].detach().cpu().numpy().astype(float),
        float(last.bias[output_index].detach().cpu().item()),
        center,
        scale,
        response_center,
        response_scale,
    )


def _from_parameters(
    hidden_weight,
    hidden_bias,
    output_weight,
    output_bias,
    center,
    scale,
    response_center,
    response_scale,
):
    return ReLUModel(
        hidden_weight,
        hidden_bias,
        np.asarray(output_weight) * response_scale,
        float(output_bias * response_scale + response_center),
        np.asarray(center, dtype=float),
        np.asarray(scale, dtype=float),
    )


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError(
            "neural workforce models require the workforce-neural dependencies"
        ) from error
    return torch, nn
