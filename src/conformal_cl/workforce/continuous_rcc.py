"""Literal RCC queue logic for a continuous two-dimensional staffing domain."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from conformal_cl.groups import Group

from .models import LinearModel, ReLUModel


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

    if model.quadratic or model.dedicated_knots or len(model.coefficient) != 3:
        raise ValueError("continuous RCC currently requires an affine model")
    slope = np.asarray(model.coefficient[1:], dtype=float) / model.scale
    intercept = float(model.coefficient[0] - slope @ model.center)
    return intercept, slope


def piecewise_affine_parameters(
    model: LinearModel,
) -> list[tuple[float, float, np.ndarray]]:
    """Return ``right endpoint, intercept, slope`` for each affine spline piece."""

    if model.quadratic:
        raise ValueError("continuous RCC does not support quadratic fitted models")
    knots = np.asarray(model.dedicated_knots, dtype=float)
    if len(model.coefficient) != 3 + len(knots):
        raise ValueError("model coefficients and spline knots do not align")
    breakpoints = model.center[0] + model.scale[0] * knots
    pieces = []
    for piece in range(len(knots) + 1):
        active = np.arange(len(knots)) < piece
        dedicated_weight = float(
            model.coefficient[1] + np.sum(model.coefficient[3:][active])
        )
        slope = np.asarray(
            [dedicated_weight / model.scale[0], model.coefficient[2] / model.scale[1]]
        )
        intercept = float(
            model.coefficient[0]
            - model.coefficient[1] * model.center[0] / model.scale[0]
            - model.coefficient[2] * model.center[1] / model.scale[1]
        )
        if piece:
            intercept += float(
                np.sum(
                    model.coefficient[3 : 3 + piece]
                    * (-model.center[0] / model.scale[0] - knots[:piece])
                )
            )
        right = float(breakpoints[piece]) if piece < len(knots) else np.inf
        pieces.append((right, intercept, slope))
    return pieces


def model_pieces(
    model: LinearModel | ReLUModel, vertices: np.ndarray
) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """Intersect a polygon with every affine piece of a fitted model."""

    if isinstance(model, ReLUModel):
        return _relu_model_pieces(model, vertices)
    return _spline_model_pieces(model, vertices)


def _spline_model_pieces(
    model: LinearModel, vertices: np.ndarray
) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """Intersect a polygon with every affine piece of a fitted spline."""

    remaining = np.asarray(vertices, dtype=float).reshape(-1, 2)
    result = []
    for right, intercept, slope in piecewise_affine_parameters(model):
        if len(remaining) == 0:
            break
        if np.isfinite(right):
            piece = clip_polygon(remaining, np.asarray([1.0, 0.0]), right)
            remaining = clip_polygon(remaining, np.asarray([-1.0, 0.0]), -right)
        else:
            piece = remaining
            remaining = np.empty((0, 2))
        if len(piece):
            result.append((piece, intercept, slope))
    return result


def _relu_model_pieces(
    model: ReLUModel, vertices: np.ndarray
) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """Enumerate a shallow ReLU network's activation regions in a polygon."""

    polygons = [np.asarray(vertices, dtype=float).reshape(-1, 2)]
    normals = model.hidden_weight / model.scale
    offsets = model.hidden_bias - normals @ model.center
    for normal, offset in zip(normals, offsets, strict=True):
        split = []
        for polygon in polygons:
            inactive = clip_polygon(polygon, normal, -offset)
            active = clip_polygon(polygon, -normal, offset)
            if _has_area(inactive):
                split.append(inactive)
            if _has_area(active):
                split.append(active)
        polygons = split
        if not polygons:
            break

    result = []
    for polygon in polygons:
        representative = np.mean(polygon, axis=0)
        active = normals @ representative + offsets > 0.0
        slope = model.output_weight[active] @ normals[active]
        intercept = float(
            model.output_bias + model.output_weight[active] @ offsets[active]
        )
        result.append((polygon, intercept, np.asarray(slope, dtype=float)))
    return result


def _has_area(vertices: np.ndarray, tolerance: float = 1e-12) -> bool:
    if len(vertices) < 3:
        return False
    x = vertices[:, 0]
    y = vertices[:, 1]
    area = 0.5 * abs(float(x @ np.roll(y, 1) - y @ np.roll(x, 1)))
    return area > tolerance


def model_extrema(
    model: LinearModel | ReLUModel, vertices: np.ndarray
) -> tuple[float, np.ndarray, float, np.ndarray]:
    """Return the exact minimum/argmin and maximum/argmax over a polygon."""

    candidates = []
    for piece, intercept, slope in model_pieces(model, vertices):
        candidates.extend((float(intercept + slope @ x), x) for x in piece)
    if not candidates:
        raise ValueError("cannot optimize a model over an empty polygon")
    minimum = min(candidates, key=lambda item: (item[0], *item[1]))
    maximum = max(candidates, key=lambda item: (item[0], *item[1]))
    return minimum[0], minimum[1].copy(), maximum[0], maximum[1].copy()


def minimize_piecewise_constraint(
    model: LinearModel | ReLUModel,
    correction: float,
    vertices: np.ndarray,
    objective_slope: np.ndarray,
) -> tuple[np.ndarray, float] | None:
    """Minimize a linear cost subject to a piecewise-affine upper bound."""

    best = None
    for piece, intercept, slope in model_pieces(model, vertices):
        feasible = clip_polygon(piece, slope, -intercept - correction)
        candidate = minimize_over_vertices(feasible, objective_slope)
        if candidate is None:
            continue
        decision, objective = candidate
        proposal = (objective, decision[0], decision[1])
        if best is None or proposal < best[0]:
            best = (proposal, decision)
    if best is None:
        return None
    return best[1], float(best[0][0])


def run_continuous_rcc(
    lower_model: LinearModel | ReLUModel,
    upper_model: LinearModel | ReLUModel,
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

    The fitted interval endpoints are affine or piecewise-affine splines, so all
    regional extrema and final linear-objective solves are exact vertex
    calculations over continuous polygons. Partition construction and local
    calibration deliberately use the same calibration sample, matching the
    original adaptive algorithm.
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

        minimum_value, minimum_x, maximum_value, maximum_x = model_extrema(
            upper_model, vertices
        )
        lower_minimum, _, _, _ = model_extrema(lower_model, vertices)
        maximum_value += region.qhat
        minimum_value += region.qhat

        if maximum_value <= feasibility_tolerance:
            safe.append(region)
            continue
        if lower_minimum - region.qhat >= -feasibility_tolerance:
            discarded += 1
            continue

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
                candidate = minimize_piecewise_constraint(
                    upper_model,
                    region.qhat,
                    vertices,
                    objective_slope,
                )
            else:
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
