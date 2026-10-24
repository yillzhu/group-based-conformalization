"""Small dependency-free regression models for the workforce experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearModel:
    coefficient: np.ndarray
    center: np.ndarray
    scale: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        return _features(x, self.center, self.scale) @ self.coefficient


def fit_mean(x: np.ndarray, y: np.ndarray) -> LinearModel:
    """Fit the deliberately low-capacity point-prediction baseline."""

    matrix, center, scale = _training_features(x)
    coefficient = np.linalg.lstsq(matrix, _response(y, len(matrix)), rcond=None)[0]
    return LinearModel(coefficient, center, scale)


def fit_quantile(
    x: np.ndarray,
    y: np.ndarray,
    quantile: float,
    *,
    iterations: int = 2500,
    learning_rate: float = 0.04,
) -> LinearModel:
    """Fit linear quantile regression with deterministic Adam iterations."""

    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    matrix, center, scale = _training_features(x)
    response = _response(y, len(matrix))
    coefficient = np.linalg.lstsq(matrix, response, rcond=None)[0]
    first = np.zeros_like(coefficient)
    second = np.zeros_like(coefficient)

    for step in range(1, iterations + 1):
        residual = response - matrix @ coefficient
        derivative = np.where(residual >= 0.0, -quantile, 1.0 - quantile)
        gradient = matrix.T @ derivative / len(matrix)
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient**2
        first_hat = first / (1.0 - 0.9**step)
        second_hat = second / (1.0 - 0.999**step)
        coefficient -= learning_rate * first_hat / (np.sqrt(second_hat) + 1e-8)

    return LinearModel(coefficient, center, scale)


def pinball_loss(y: np.ndarray, prediction: np.ndarray, quantile: float) -> float:
    residual = np.asarray(y) - np.asarray(prediction)
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1) * residual)))


def _training_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all():
        raise ValueError("x must be a finite array with shape (n, 2)")
    center = (x.min(axis=0) + x.max(axis=0)) / 2.0
    scale = (x.max(axis=0) - x.min(axis=0)) / 2.0
    if np.any(scale <= 0.0):
        raise ValueError("each decision feature must vary in the training data")
    return _features(x, center, scale), center, scale


def _features(x: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all():
        raise ValueError("x must be a finite array with shape (n, 2)")
    normalized = (x - center) / scale
    # Linearity is intentional: the experiment asks whether localized conformal
    # corrections can repair practically plausible model misspecification.
    return np.column_stack((np.ones(len(x)), normalized))


def _response(y: np.ndarray, expected: int) -> np.ndarray:
    response = np.asarray(y, dtype=float).reshape(-1)
    if len(response) != expected or not np.isfinite(response).all():
        raise ValueError("y must be finite and have one entry per row of x")
    return response
