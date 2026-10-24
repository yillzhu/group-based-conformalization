"""Independent partition design and one-sided conformal calibration."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from conformal_cl.order_stats import conformal_order_statistic


@dataclass(frozen=True)
class Cell:
    lower: np.ndarray
    upper: np.ndarray

    def contains(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.all((x >= self.lower) & (x <= self.upper), axis=1)


@dataclass(frozen=True)
class CalibratedPartition:
    cells: tuple[Cell, ...]
    corrections: np.ndarray
    calibration_counts: np.ndarray

    def correction(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        result = np.full(len(x), np.nan)
        for cell, value in zip(self.cells, self.corrections, strict=True):
            member = cell.contains(x)
            if np.any(member & np.isfinite(result)):
                raise RuntimeError("partition cells overlap")
            result[member] = value
        if not np.isfinite(result).all():
            raise ValueError("some decisions fall outside the calibrated partition")
        return result


def global_correction(scores: np.ndarray, alpha: float) -> float:
    return conformal_order_statistic(scores, alpha=alpha)


def design_partition(
    x_design: np.ndarray,
    scores: np.ndarray,
    candidates: np.ndarray,
    upper_prediction: np.ndarray,
    *,
    alpha: float,
    max_splits: int = 6,
    min_design_size: int = 100,
) -> tuple[Cell, ...]:
    """Design an axis-aligned partition without using final calibration data.

    Only cells whose fitted, design-corrected upper surface crosses zero are
    eligible.  Within those cells, choose the split with the largest difference
    in child score quantiles.  This targets heterogeneous error near the learned
    feasibility boundary rather than partitioning the whole decision space.
    """

    x_design = np.asarray(x_design, dtype=float)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    candidates = np.asarray(candidates, dtype=float)
    upper_prediction = np.asarray(upper_prediction, dtype=float).reshape(-1)
    if len(x_design) != len(scores):
        raise ValueError("design points and scores must have the same length")
    if len(candidates) != len(upper_prediction):
        raise ValueError("candidates and upper_prediction must have the same length")
    if max_splits < 0 or min_design_size < 2:
        raise ValueError("invalid partition-size parameters")

    cells = [Cell(candidates.min(axis=0), candidates.max(axis=0))]
    for _ in range(max_splits):
        best = None
        for cell_index, cell in enumerate(cells):
            design_member = cell.contains(x_design)
            candidate_member = cell.contains(candidates)
            if design_member.sum() < 2 * min_design_size:
                continue
            local_scores = scores[design_member]
            correction = _empirical_upper_quantile(local_scores, 1.0 - alpha)
            local_upper = upper_prediction[candidate_member] + correction
            if not (np.any(local_upper <= 0.0) and np.any(local_upper > 0.0)):
                continue

            local_x = x_design[design_member]
            for axis in range(2):
                values = np.unique(candidates[candidate_member, axis])
                for threshold in (values[:-1] + values[1:]) / 2.0:
                    left = local_x[:, axis] <= threshold
                    if left.sum() < min_design_size or (~left).sum() < min_design_size:
                        continue
                    q_left = _empirical_upper_quantile(
                        local_scores[left], 1.0 - alpha
                    )
                    q_right = _empirical_upper_quantile(
                        local_scores[~left], 1.0 - alpha
                    )
                    gain = abs(q_left - q_right)
                    proposal = (gain, -cell_index, -axis, -threshold)
                    if best is None or proposal > best[0]:
                        best = (proposal, cell_index, axis, threshold)
        if best is None:
            break
        _, cell_index, axis, threshold = best
        parent = cells.pop(cell_index)
        left_upper = parent.upper.copy()
        left_upper[axis] = threshold
        right_lower = parent.lower.copy()
        right_lower[axis] = np.nextafter(threshold, np.inf)
        cells.extend(
            [Cell(parent.lower.copy(), left_upper), Cell(right_lower, parent.upper.copy())]
        )
    return tuple(cells)


def calibrate_partition(
    cells: tuple[Cell, ...],
    x_calibration: np.ndarray,
    scores: np.ndarray,
    *,
    alpha: float,
) -> CalibratedPartition:
    """Calibrate every fixed cell on data untouched by partition design."""

    x_calibration = np.asarray(x_calibration, dtype=float)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(x_calibration) != len(scores):
        raise ValueError("calibration points and scores must have the same length")
    corrections = []
    for cell in cells:
        member = cell.contains(x_calibration)
        if member.sum() < math.ceil(1.0 / alpha):
            raise ValueError("a partition cell has insufficient final calibration data")
        corrections.append(conformal_order_statistic(scores[member], alpha=alpha))
    counts = np.asarray([cell.contains(x_calibration).sum() for cell in cells])
    return CalibratedPartition(cells, np.asarray(corrections), counts)


def _empirical_upper_quantile(values: np.ndarray, probability: float) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    index = min(len(values) - 1, max(0, math.ceil(probability * len(values)) - 1))
    return float(values[index])
