"""Literal RCC queue logic for a continuous two-dimensional staffing domain."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from conformal_cl.groups import Group

from .models import LinearModel


@dataclass(frozen=True)
class ContinuousRCCResult:
    status: str
    decision: np.ndarray | None
    objective: float | None
    route: str | None
    correction: float | None
    splits: int
    safe_regions: int
    unresolved_regions: int
    discarded_regions: int


def affine_parameters(model: LinearModel) -> tuple[float, np.ndarray]:
    """Return ``intercept, slope`` for a nonquadratic fitted model."""

    if model.quadratic or len(model.coefficient) != 3:
        raise ValueError("continuous RCC currently requires an affine model")
    slope = np.asarray(model.coefficient[1:], dtype=float) / model.scale
    intercept = float(model.coefficient[0] - slope @ model.center)
    return intercept, slope


def run_continuous_rcc(
    lower_model: LinearModel,
    upper_model: LinearModel,
    calibration_x: np.ndarray,
    calibration_y: np.ndarray,
    *,
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    objective_slope: np.ndarray = np.asarray([2.0, 2.4]),
    alpha: float = 0.1,
    delta: float = 0.1,
    max_splits: int = 64,
    feasibility_tolerance: float = 1e-9,
) -> ContinuousRCCResult:
    """Run the Chapter 4 RCC algorithm over a continuous rectangular domain.

    The fitted interval endpoints are affine, so all regional extrema and final
    linear-objective solves are exact vertex calculations over continuous
    polygons.  Partition construction and local calibration deliberately use
    the same calibration sample, matching the original adaptive algorithm.
    """

    calibration_x = np.asarray(calibration_x, dtype=float)
    calibration_y = np.asarray(calibration_y, dtype=float).reshape(-1)
    lower = np.asarray(domain_lower, dtype=float).reshape(-1)
    upper = np.asarray(domain_upper, dtype=float).reshape(-1)
    objective_slope = np.asarray(objective_slope, dtype=float).reshape(-1)
    if calibration_x.ndim != 2 or calibration_x.shape[1] != 2:
        raise ValueError("calibration_x must have shape (n, 2)")
    if len(calibration_x) != len(calibration_y):
        raise ValueError("calibration inputs and responses must align")
    if lower.shape != (2,) or upper.shape != (2,) or np.any(lower >= upper):
        raise ValueError("domain bounds must define a nonempty 2D rectangle")
    if objective_slope.shape != (2,) or max_splits < 0:
        raise ValueError("invalid objective or split limit")

    lower_intercept, lower_slope = affine_parameters(lower_model)
    upper_intercept, upper_slope = affine_parameters(upper_model)
    lower_prediction = lower_model.predict(calibration_x)
    upper_prediction = upper_model.predict(calibration_x)
    scores = np.maximum(
        calibration_y - upper_prediction,
        lower_prediction - calibration_y,
    )

    root = Group(bounds={})
    root.conformalize(calibration_x, scores, alpha=alpha, delta=delta)
    queue: deque[Group] = deque()
    safe: list[Group] = []
    unresolved: list[Group] = []
    discarded = 0
    if root.Xhat:
        queue.append(root)
    else:
        unresolved.append(root)

    splits = 0
    while queue and splits < max_splits:
        region = queue.popleft()
        vertices = region_vertices(region, lower, upper)
        if len(vertices) == 0:
            discarded += 1
            continue

        upper_values = upper_intercept + vertices @ upper_slope + region.qhat
        lower_values = lower_intercept + vertices @ lower_slope - region.qhat
        maximum_index = int(np.argmax(upper_values))
        minimum_index = int(np.argmin(upper_values))
        maximum_value = float(upper_values[maximum_index])
        minimum_value = float(upper_values[minimum_index])

        if maximum_value <= feasibility_tolerance:
            safe.append(region)
            continue
        if float(np.min(lower_values)) >= -feasibility_tolerance:
            discarded += 1
            continue

        maximum_x = vertices[maximum_index]
        minimum_x = vertices[minimum_index]
        normal = maximum_x - minimum_x
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm <= feasibility_tolerance:
            unresolved.append(region)
            continue
        normal /= norm
        if minimum_value <= -feasibility_tolerance and not np.isclose(
            maximum_value,
            minimum_value,
            atol=feasibility_tolerance,
            rtol=0.0,
        ):
            split_point = minimum_x - minimum_value * (
                maximum_x - minimum_x
            ) / (maximum_value - minimum_value)
        else:
            split_point = (minimum_x + maximum_x) / 2.0
        coefficients = np.asarray(
            [normal[0], normal[1], float(normal @ split_point)]
        )
        if tuple(coefficients) in region.bounds:
            unresolved.append(region)
            continue

        first, second = region.split(
            coefficients,
            calibration_x,
            scores,
            alpha=alpha,
            delta=delta,
        )
        for child in (first, second):
            if len(region_vertices(child, lower, upper)) == 0:
                discarded += 1
            elif child.Xhat:
                queue.append(child)
            else:
                unresolved.append(child)
        splits += 1

    unresolved.extend(queue)

    best = None
    for route, regions in (("safe", safe), ("unresolved", unresolved)):
        for region in regions:
            vertices = region_vertices(region, lower, upper)
            if route == "unresolved":
                vertices = clip_polygon(
                    vertices,
                    upper_slope,
                    -upper_intercept - region.qhat,
                )
            candidate = minimize_over_vertices(vertices, objective_slope)
            if candidate is None:
                continue
            decision, objective = candidate
            proposal = (objective, route != "safe", decision[0], decision[1])
            if best is None or proposal < best[0]:
                best = (proposal, decision, route, float(region.qhat))

    if best is None:
        return ContinuousRCCResult(
            "no_solution",
            None,
            None,
            None,
            None,
            splits,
            len(safe),
            len(unresolved),
            discarded,
        )
    proposal, decision, route, correction = best
    return ContinuousRCCResult(
        "selected",
        decision,
        float(proposal[0]),
        route,
        correction,
        splits,
        len(safe),
        len(unresolved),
        discarded,
    )


def region_vertices(
    region: Group, domain_lower: np.ndarray, domain_upper: np.ndarray
) -> np.ndarray:
    """Return the vertices of a group intersected with a rectangular domain."""

    lo = np.asarray(domain_lower, dtype=float)
    hi = np.asarray(domain_upper, dtype=float)
    polygon = np.asarray(
        [[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]],
        dtype=float,
    )
    for coefficients, direction in region.bounds.items():
        coefficients = np.asarray(coefficients, dtype=float)
        polygon = clip_polygon(
            polygon,
            direction * coefficients[:2],
            direction * coefficients[2],
        )
        if len(polygon) == 0:
            break
    return polygon


def clip_polygon(
    vertices: np.ndarray,
    normal: np.ndarray,
    threshold: float,
    *,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Intersect a convex polygon with ``normal @ x <= threshold``."""

    vertices = np.asarray(vertices, dtype=float).reshape(-1, 2)
    normal = np.asarray(normal, dtype=float).reshape(2)
    if len(vertices) == 0:
        return vertices
    output = []
    previous = vertices[-1]
    previous_value = float(normal @ previous - threshold)
    previous_inside = previous_value <= tolerance
    for current in vertices:
        current_value = float(normal @ current - threshold)
        current_inside = current_value <= tolerance
        if current_inside != previous_inside:
            direction = current - previous
            denominator = float(normal @ direction)
            if abs(denominator) > tolerance:
                fraction = float((threshold - normal @ previous) / denominator)
                output.append(previous + fraction * direction)
        if current_inside:
            output.append(current)
        previous = current
        previous_inside = current_inside
    if not output:
        return np.empty((0, 2))
    result = np.asarray(output)
    keep = np.ones(len(result), dtype=bool)
    for index in range(1, len(result)):
        keep[index] = not np.allclose(result[index], result[index - 1], atol=tolerance)
    result = result[keep]
    if len(result) > 1 and np.allclose(result[0], result[-1], atol=tolerance):
        result = result[:-1]
    return result


def minimize_over_vertices(
    vertices: np.ndarray, objective_slope: np.ndarray
) -> tuple[np.ndarray, float] | None:
    if len(vertices) == 0:
        return None
    values = np.asarray(vertices) @ np.asarray(objective_slope)
    index = int(np.argmin(values))
    return np.asarray(vertices[index]).copy(), float(values[index])
