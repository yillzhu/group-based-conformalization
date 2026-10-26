"""Finite-candidate analogue of the Chapter 4 RCC partition queue."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from conformal_cl.groups import Group
from conformal_cl.order_stats import (
    InsufficientCalibrationError,
    conformal_order_statistic,
)


@dataclass(frozen=True)
class CandidateRCCResult:
    status: str
    index: int | None
    correction: float | None
    splits: int
    terminal_regions: int
    discarded_regions: int


def run_candidate_rcc(
    design_x: np.ndarray,
    design_scores: np.ndarray,
    calibration_x: np.ndarray,
    calibration_scores: np.ndarray,
    candidate_x: np.ndarray,
    candidate_prediction: np.ndarray,
    candidate_cost: np.ndarray,
    *,
    alpha: float = .1,
    delta: float = .1,
    max_splits: int = 32,
) -> CandidateRCCResult:
    """Partition a candidate set with RCC geometry, then independently recalibrate.

    Extrema are exact only over the supplied candidate set, so this is an audit
    implementation rather than a continuous optimization algorithm.  Unlike the
    historical implementation, partition design and final calibration are
    independent.
    """

    design_x = np.asarray(design_x, dtype=float)
    calibration_x = np.asarray(calibration_x, dtype=float)
    candidate_x = np.asarray(candidate_x, dtype=float)
    design_scores = np.asarray(design_scores, dtype=float).reshape(-1)
    calibration_scores = np.asarray(calibration_scores, dtype=float).reshape(-1)
    prediction = np.asarray(candidate_prediction, dtype=float).reshape(-1)
    cost = np.asarray(candidate_cost, dtype=float).reshape(-1)
    if design_x.ndim != 2 or calibration_x.ndim != 2 or candidate_x.ndim != 2:
        raise ValueError("RCC inputs must be two-dimensional")
    if not design_x.shape[1] == calibration_x.shape[1] == candidate_x.shape[1]:
        raise ValueError("RCC feature dimensions must agree")
    if len(design_x) != len(design_scores):
        raise ValueError("design inputs and scores must align")
    if len(calibration_x) != len(calibration_scores):
        raise ValueError("calibration inputs and scores must align")
    if len(candidate_x) != len(prediction) or len(candidate_x) != len(cost):
        raise ValueError("candidate arrays must align")

    root = Group()
    root.conformalize(design_x, design_scores, alpha=alpha, delta=delta)
    queue = deque([root]) if root.Xhat else deque()
    terminal = [] if root.Xhat else [root]
    discarded = 0
    splits = 0
    while queue and splits < max_splits:
        region = queue.popleft()
        mask = _membership(region, candidate_x)
        indices = np.flatnonzero(mask)
        if not len(indices):
            discarded += 1
            continue
        values = prediction[indices] + region.qhat
        if np.max(values) <= 0.:
            terminal.append(region)
            continue
        if np.min(values) > 0.:
            discarded += 1
            continue
        low_index = indices[np.argmin(values)]
        high_index = indices[np.argmax(values)]
        low_x = candidate_x[low_index]
        high_x = candidate_x[high_index]
        normal = high_x - low_x
        norm = np.linalg.norm(normal)
        if norm <= 1e-12:
            terminal.append(region)
            continue
        normal /= norm
        low_value = prediction[low_index] + region.qhat
        high_value = prediction[high_index] + region.qhat
        split_point = low_x - low_value * (high_x - low_x) / (
            high_value - low_value
        )
        coefficients = np.r_[normal, normal @ split_point]
        if tuple(coefficients) in region.bounds:
            terminal.append(region)
            continue
        children = region.split(
            coefficients, design_x, design_scores, alpha=alpha, delta=delta
        )
        for child in children:
            if child.Xhat:
                queue.append(child)
            else:
                terminal.append(child)
        splits += 1
    terminal.extend(queue)

    best = None
    for region in terminal:
        calibration_mask = _membership(region, calibration_x)
        try:
            correction = conformal_order_statistic(
                calibration_scores[calibration_mask], alpha=alpha
            )
        except InsufficientCalibrationError:
            continue
        candidate_mask = _membership(region, candidate_x)
        feasible = candidate_mask & (prediction + correction <= 0.)
        indices = np.flatnonzero(feasible)
        if not len(indices):
            continue
        index = int(indices[np.argmin(cost[indices])])
        proposal = (float(cost[index]), index, float(correction))
        if best is None or proposal < best:
            best = proposal
    if best is None:
        return CandidateRCCResult(
            "no_candidate", None, None, splits, len(terminal), discarded
        )
    return CandidateRCCResult(
        "selected", best[1], best[2], splits, len(terminal), discarded
    )


def _membership(region: Group, x: np.ndarray) -> np.ndarray:
    mask = np.ones(len(x), dtype=bool)
    for coefficients, direction in region.bounds.items():
        coefficients = np.asarray(coefficients, dtype=float)
        mask &= direction * (x @ coefficients[:-1]) <= direction * coefficients[-1]
    return mask
