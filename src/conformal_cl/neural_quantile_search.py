"""Small smooth neural quantile model for structure-free search proposals."""

from __future__ import annotations

import numpy as np


class NeuralQuantileSearchModel:
    """Two-tanh-layer quantile regressor, warm-started after each audit.

    Inputs and response are standardized using *initial* search data only.
    Later audit observations update the weights without changing coordinates.
    Predictions are proposal scores, not calibrated confidence bounds.
    """

    def __init__(self, x, y, quantile, *, seed=0, hidden_units=24):
        import torch
        from torch import nn

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if x.ndim != 2 or len(x) != len(y) or x.shape[1] < 1:
            raise ValueError("x and y must be aligned")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("training data must be finite")
        if not 0 < quantile < 1 or hidden_units < 1:
            raise ValueError("invalid neural quantile configuration")
        self.torch = torch
        self.quantile = quantile
        self.x_center = x.mean(axis=0)
        self.x_scale = np.maximum(x.std(axis=0), 1e-6)
        self.y_center = float(y.mean())
        self.y_scale = max(float(y.std()), 1e-6)
        torch.manual_seed(seed)
        self.network = nn.Sequential(
            nn.Linear(x.shape[1], hidden_units), nn.Tanh(),
            nn.Linear(hidden_units, hidden_units), nn.Tanh(),
            nn.Linear(hidden_units, 1),
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.01,
                                          weight_decay=1e-4)

    def fit(self, x, y, *, epochs):
        if epochs < 1:
            raise ValueError("epochs must be positive")
        torch = self.torch
        inputs = torch.as_tensor(
            (np.asarray(x, dtype=float) - self.x_center) / self.x_scale,
            dtype=torch.float32,
        )
        response = torch.as_tensor(
            (np.asarray(y, dtype=float).reshape(-1) - self.y_center) / self.y_scale,
            dtype=torch.float32,
        )
        for _ in range(epochs):
            self.optimizer.zero_grad()
            residual = response - self.network(inputs).reshape(-1)
            loss = torch.maximum(self.quantile * residual,
                                 (self.quantile - 1) * residual).mean()
            loss.backward()
            self.optimizer.step()
        return self

    def predict(self, x):
        inputs = self.torch.as_tensor(
            (np.asarray(x, dtype=float) - self.x_center) / self.x_scale,
            dtype=self.torch.float32,
        )
        with self.torch.no_grad():
            prediction = self.network(inputs).reshape(-1).cpu().numpy()
        return self.y_center + self.y_scale * prediction
