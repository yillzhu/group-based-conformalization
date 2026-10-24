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
class PolyhedralCell:
    """Intersection of oriented halfspaces in two decision dimensions."""

    bounds: tuple[tuple[float, float, float, int], ...] = ()

    def contains(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        member = np.ones(len(x), dtype=bool)
        for first, second, threshold, direction in self.bounds:
            projection = first * x[:, 0] + second * x[:, 1]
            member &= direction * (projection - threshold) <= 1e-12
        return member


@dataclass(frozen=True)
class CalibratedPartition:
    cells: tuple[Cell | PolyhedralCell, ...]
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


@dataclass(frozen=True)
class CalibratedStrata:
    thresholds: np.ndarray
    corrections: np.ndarray
    calibration_counts: np.ndarray

    def correction(self, prediction: np.ndarray) -> np.ndarray:
        prediction = np.asarray(prediction, dtype=float).reshape(-1)
        labels = np.searchsorted(self.thresholds, prediction, side="right")
        return self.corrections[labels]


def global_correction(scores: np.ndarray, alpha: float) -> float:
    return conformal_order_statistic(scores, alpha=alpha)


def calibrate_prediction_strata(
    candidate_prediction: np.ndarray,
    calibration_prediction: np.ndarray,
    calibration_scores: np.ndarray,
    *,
    alpha: float,
    strata: int = 8,
) -> CalibratedStrata:
    """Calibrate fixed bands of fitted risk using untouched observations.

    Thresholds depend only on the fitted model and candidate decisions, not on
    partition-design or final-calibration responses.  This is a diagnostic for
    whether groups aligned with predicted risk outperform geometric rectangles.
    """

    candidates = np.sort(np.asarray(candidate_prediction, dtype=float).reshape(-1))
    cal_prediction = np.asarray(calibration_prediction, dtype=float).reshape(-1)
    scores = np.asarray(calibration_scores, dtype=float).reshape(-1)
    if len(cal_prediction) != len(scores):
        raise ValueError("calibration predictions and scores must have equal lengths")
    if strata < 1 or strata > len(candidates):
        raise ValueError("strata must be between one and the candidate count")
    chunks = np.array_split(candidates, strata)
    thresholds = np.asarray(
        [(chunks[index][-1] + chunks[index + 1][0]) / 2.0 for index in range(strata - 1)]
    )
    labels = np.searchsorted(thresholds, cal_prediction, side="right")
    corrections = []
    counts = []
    for label in range(strata):
        member = labels == label
        counts.append(int(member.sum()))
        corrections.append(conformal_order_statistic(scores[member], alpha=alpha))
    return CalibratedStrata(
        thresholds, np.asarray(corrections), np.asarray(counts, dtype=int)
    )


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
            raw_upper = upper_prediction[candidate_member]
            local_upper = raw_upper + correction
            corrected_crossing = np.any(local_upper <= 0.0) and np.any(
                local_upper > 0.0
            )
            raw_crossing = np.any(raw_upper <= 0.0) and np.any(raw_upper > 0.0)
            # A large root correction can make every point look infeasible even
            # when heterogeneous residuals would yield feasible child regions.
            # The two-sided chapter algorithm keeps splitting in the analogous
            # case when the lower endpoint still indicates ambiguity.  For this
            # one-sided experiment, the raw fitted boundary supplies that signal.
            if not (corrected_crossing or raw_crossing):
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


def design_oblique_partition(
    x_design: np.ndarray,
    scores: np.ndarray,
    candidates: np.ndarray,
    upper_prediction: np.ndarray,
    *,
    alpha: float,
    max_splits: int = 6,
    min_design_size: int = 100,
    directions: int = 13,
) -> tuple[PolyhedralCell, ...]:
    """Design response-adaptive halfspace cells, including diagonal cuts."""

    x_design = np.asarray(x_design, dtype=float)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    candidates = np.asarray(candidates, dtype=float)
    upper_prediction = np.asarray(upper_prediction, dtype=float).reshape(-1)
    if len(x_design) != len(scores) or len(candidates) != len(upper_prediction):
        raise ValueError("features, scores, candidates, and predictions must align")
    if directions < 2:
        raise ValueError("directions must be at least two")
    angles = np.linspace(0.0, np.pi / 2.0, directions)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    cells = [PolyhedralCell()]

    for _ in range(max_splits):
        best = None
        for cell_index, cell in enumerate(cells):
            design_member = cell.contains(x_design)
            candidate_member = cell.contains(candidates)
            if design_member.sum() < 2 * min_design_size:
                continue
            local_scores = scores[design_member]
            correction = _empirical_upper_quantile(local_scores, 1.0 - alpha)
            raw_upper = upper_prediction[candidate_member]
            corrected_upper = raw_upper + correction
            corrected_crossing = np.any(corrected_upper <= 0.0) and np.any(
                corrected_upper > 0.0
            )
            raw_crossing = np.any(raw_upper <= 0.0) and np.any(raw_upper > 0.0)
            if not (corrected_crossing or raw_crossing):
                continue

            local_x = x_design[design_member]
            candidate_x = candidates[candidate_member]
            for normal_index, normal in enumerate(normals):
                candidate_projection = np.unique(candidate_x @ normal)
                thresholds = (
                    candidate_projection[:-1] + candidate_projection[1:]
                ) / 2.0
                design_projection = local_x @ normal
                for threshold in thresholds:
                    left = design_projection <= threshold
                    if left.sum() < min_design_size or (~left).sum() < min_design_size:
                        continue
                    q_left = _empirical_upper_quantile(
                        local_scores[left], 1.0 - alpha
                    )
                    q_right = _empirical_upper_quantile(
                        local_scores[~left], 1.0 - alpha
                    )
                    gain = abs(q_left - q_right)
                    proposal = (gain, -cell_index, -normal_index, -threshold)
                    if best is None or proposal > best[0]:
                        best = (
                            proposal,
                            cell_index,
                            normal,
                            float(threshold),
                        )
        if best is None:
            break
        _, cell_index, normal, threshold = best
        parent = cells.pop(cell_index)
        first, second = map(float, normal)
        left_bound = (first, second, threshold, 1)
        right_bound = (first, second, threshold, -1)
        cells.extend(
            [
                PolyhedralCell((*parent.bounds, left_bound)),
                PolyhedralCell((*parent.bounds, right_bound)),
            ]
        )
    return tuple(cells)


def design_optimization_partition(
    x_design: np.ndarray,
    scores: np.ndarray,
    candidates: np.ndarray,
    lower_prediction: np.ndarray,
    upper_prediction: np.ndarray,
    *,
    alpha: float,
    max_splits: int = 6,
    min_design_size: int = 100,
    feasibility_tolerance: float = 1e-12,
) -> tuple[PolyhedralCell, ...]:
    """Construct the optimization-driven partition used by one-sided RCC.

    This is the discrete-grid analogue of the splitting rule in Chapter 4.
    Within a cell, a design-sample correction conformalizes the fitted upper
    endpoint.  Cells whose corrected endpoint is uniformly safe or uniformly
    unsafe are terminal.  An ambiguous cell is split by a hyperplane normal to
    the segment joining the endpoint minimizer and maximizer, at the estimated
    zero crossing along that segment (or at its midpoint if the corrected
    surface does not cross zero).

    Design responses affect the temporary corrections that guide the search,
    but not the final cell corrections.  Those must be computed later from an
    independent calibration sample with :func:`calibrate_partition`.
    """

    x_design = np.asarray(x_design, dtype=float)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    candidates = np.asarray(candidates, dtype=float)
    lower_prediction = np.asarray(lower_prediction, dtype=float).reshape(-1)
    upper_prediction = np.asarray(upper_prediction, dtype=float).reshape(-1)
    if (
        len(x_design) != len(scores)
        or len(candidates) != len(lower_prediction)
        or len(candidates) != len(upper_prediction)
    ):
        raise ValueError("features, scores, candidates, and predictions must align")
    if max_splits < 0 or min_design_size < 2:
        raise ValueError("invalid partition-size parameters")

    cells = [PolyhedralCell()]
    for _ in range(max_splits):
        split = None
        for cell_index, cell in enumerate(cells):
            design_member = cell.contains(x_design)
            candidate_indices = np.flatnonzero(cell.contains(candidates))
            if design_member.sum() < 2 * min_design_size or len(candidate_indices) < 2:
                continue

            correction = _empirical_upper_quantile(
                scores[design_member], 1.0 - alpha
            )
            local_values = upper_prediction[candidate_indices] + correction
            local_lower = lower_prediction[candidate_indices] - correction
            minimum_position = int(np.argmin(local_values))
            maximum_position = int(np.argmax(local_values))
            minimum_value = float(local_values[minimum_position])
            maximum_value = float(local_values[maximum_position])

            # As in Chapter 4, a region is terminal only when its upper endpoint
            # is uniformly safe or its lower endpoint is uniformly infeasible.
            # The lower endpoint is used to recognize ambiguity, not to enforce
            # the one-sided chance constraint.
            uniformly_safe = maximum_value <= feasibility_tolerance
            uniformly_infeasible = (
                float(np.min(local_lower)) >= -feasibility_tolerance
            )
            if uniformly_safe or uniformly_infeasible:
                continue

            minimum_x = candidates[candidate_indices[minimum_position]]
            maximum_x = candidates[candidate_indices[maximum_position]]
            normal = maximum_x - minimum_x
            norm = float(np.linalg.norm(normal))
            if not np.isfinite(norm) or norm <= feasibility_tolerance:
                continue
            normal = normal / norm

            if minimum_value <= feasibility_tolerance and not np.isclose(
                maximum_value, minimum_value, atol=feasibility_tolerance, rtol=0.0
            ):
                split_point = minimum_x - minimum_value * (
                    maximum_x - minimum_x
                ) / (maximum_value - minimum_value)
            else:
                split_point = (minimum_x + maximum_x) / 2.0
            desired_threshold = float(normal @ split_point)
            projections = np.unique(candidates[candidate_indices] @ normal)
            separated = np.diff(projections) > 4e-12
            gap_thresholds = (
                projections[:-1][separated] + projections[1:][separated]
            ) / 2.0
            if len(gap_thresholds) == 0:
                continue
            children = None
            for gap_index in np.argsort(np.abs(gap_thresholds - desired_threshold)):
                threshold = float(gap_thresholds[gap_index])
                left_bound = (float(normal[0]), float(normal[1]), threshold, 1)
                right_bound = (float(normal[0]), float(normal[1]), threshold, -1)
                left = PolyhedralCell((*cell.bounds, left_bound))
                right = PolyhedralCell((*cell.bounds, right_bound))
                if min(
                    int(left.contains(x_design).sum()),
                    int(right.contains(x_design).sum()),
                ) < min_design_size:
                    continue
                if not left.contains(candidates).any() or not right.contains(candidates).any():
                    continue
                children = (left, right)
                break
            if children is None:
                continue
            left, right = children

            # Process the most strongly ambiguous cell first.  This is only a
            # deterministic queue priority; the split itself is fixed by the
            # regional extrema above.
            ambiguity = maximum_value - minimum_value
            proposal = (ambiguity, -cell_index)
            if split is None or proposal > split[0]:
                split = (proposal, cell_index, left, right)

        if split is None:
            break
        _, cell_index, left, right = split
        cells.pop(cell_index)
        cells.extend((left, right))
    return tuple(cells)


def calibrate_partition(
    cells: tuple[Cell | PolyhedralCell, ...],
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
