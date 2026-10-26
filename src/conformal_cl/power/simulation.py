"""DC power-flow simulator and full-support dispatch sampling."""

from __future__ import annotations

import numpy as np

from .case39 import Case39


def sample_dispatches(
    case: Case39,
    size: int,
    rng: np.random.Generator,
    *,
    burn_in: int = 500,
    thinning: int = 3,
) -> np.ndarray:
    """Hit-and-run samples on ``{p: sum(p)=demand, p_min <= p <= p_max}``.

    Every relatively open subset of the dispatch polytope has positive sampling
    probability.  Samples are not claimed to be independent or exactly uniform.
    """

    if size <= 0 or burn_in < 0 or thinning <= 0:
        raise ValueError("invalid hit-and-run settings")
    if not case.p_min.sum() < case.demand < case.p_max.sum():
        raise ValueError("demand must lie strictly between aggregate bounds")
    p = case.p_min + (case.demand - case.p_min.sum()) * (
        (case.p_max - case.p_min) / (case.p_max - case.p_min).sum()
    )
    return _hit_and_run(
        p, case.p_min, case.p_max, size, rng, burn_in, thinning
    )


def sample_security_focused_dispatches(
    simulator: "DCSecuritySimulator",
    size: int,
    rng: np.random.Generator,
    *,
    utilization_limit: float = 1.0,
    reserve_fraction: float = 0.02,
    burn_in: int = 500,
    thinning: int = 3,
) -> np.ndarray:
    """Sample the dispatch polytope restricted by nominal DC line limits."""

    if utilization_limit <= 0. or not 0. <= reserve_fraction < .5:
        raise ValueError("invalid security-focused sampling settings")
    try:
        from scipy.optimize import linprog
    except ImportError as error:
        raise RuntimeError("security-focused sampling requires scipy") from error
    case = simulator.case
    lower_bound = np.maximum(case.p_min, reserve_fraction * case.p_max)
    upper_bound = np.minimum(case.p_max, (1. - reserve_fraction) * case.p_max)
    flow = simulator.nominal_flow_matrix
    generator_flow = flow @ simulator.generator_map
    load_flow = flow @ case.loads
    line_limit = utilization_limit * case.branch_rate
    a_ub = np.vstack((generator_flow, -generator_flow))
    b_ub = np.hstack((line_limit + load_flow, line_limit - load_flow))
    # Find a point well inside the line constraints by minimizing maximum
    # utilization.  This avoids initializing hit-and-run at a sharp vertex.
    a_minimax = np.column_stack((a_ub, -np.tile(line_limit, 2)))
    result = linprog(
        np.r_[np.zeros(len(case.p_max)), 1.],
        A_ub=a_minimax,
        b_ub=np.hstack((load_flow, -load_flow)),
        A_eq=np.r_[np.ones(len(case.p_max)), 0.][None, :],
        b_eq=[case.demand],
        bounds=[*zip(lower_bound, upper_bound), (0., None)],
        method="highs",
    )
    if not result.success or np.any(a_ub @ result.x[:-1] > b_ub + 1e-7):
        raise ValueError("nominal security-focused dispatch polytope is empty")
    return _hit_and_run(
        result.x[:-1], lower_bound, upper_bound, size, rng, burn_in, thinning,
        a_ub=a_ub, b_ub=b_ub,
    )


def _hit_and_run(
    initial, lower_bound, upper_bound, size, rng, burn_in, thinning,
    *, a_ub=None, b_ub=None,
):
    p = np.asarray(initial, dtype=float).copy()
    samples = np.empty((size, len(p)))
    saved = 0
    total_steps = burn_in + size * thinning
    for step in range(total_steps):
        direction = rng.normal(size=len(p))
        direction -= direction.mean()
        positive = direction > 1e-14
        negative = direction < -1e-14
        lower = -np.inf
        upper = np.inf
        if np.any(positive):
            lower = max(lower, np.max((lower_bound[positive] - p[positive]) / direction[positive]))
            upper = min(upper, np.min((upper_bound[positive] - p[positive]) / direction[positive]))
        if np.any(negative):
            lower = max(lower, np.max((upper_bound[negative] - p[negative]) / direction[negative]))
            upper = min(upper, np.min((lower_bound[negative] - p[negative]) / direction[negative]))
        if a_ub is not None:
            slack = b_ub - a_ub @ p
            directional = a_ub @ direction
            moves_up = directional > 1e-14
            moves_down = directional < -1e-14
            if np.any(moves_up):
                upper = min(upper, np.min(slack[moves_up] / directional[moves_up]))
            if np.any(moves_down):
                lower = max(lower, np.max(slack[moves_down] / directional[moves_down]))
        p = p + rng.uniform(lower, upper) * direction
        if step >= burn_in and (step - burn_in) % thinning == 0:
            samples[saved] = p
            saved += 1
    return samples


class DCSecuritySimulator:
    """Evaluate dispatches under correlated load errors and line contingencies."""

    def __init__(self, case: Case39, *, outage_probability: float = 0.05):
        if not 0. <= outage_probability <= 1.:
            raise ValueError("outage_probability must lie in [0, 1]")
        self.case = case
        self.outage_probability = float(outage_probability)
        self._incidence = np.zeros((len(case.branch_x), len(case.loads)))
        rows = np.arange(len(case.branch_x))
        self._incidence[rows, case.branch_from] = 1.
        self._incidence[rows, case.branch_to] = -1.
        self._susceptance = 1. / (case.branch_x * case.branch_tap)
        self.outage_lines, matrices = self._flow_matrices()
        self._flow_matrix = np.stack([self._base_flow_matrix(), *matrices])
        self._generator_map = np.zeros((len(case.loads), len(case.p_max)))
        self._generator_map[case.generator_bus, np.arange(len(case.p_max))] = 1.
        self._participation = case.p_max / case.p_max.sum()
        positive_load = case.loads > 0.
        # Four geographic load-error regions, ordered by bus number.  The common
        # and regional terms create realistic dependence without fitting a
        # distribution to this benchmark case.
        self._load_bus = np.flatnonzero(positive_load)
        self._load_region = np.minimum(self._load_bus * 4 // len(case.loads), 3)

    @property
    def contingencies(self) -> int:
        """Number of connected single-line outages in the experiment."""

        return len(self.outage_lines)

    @property
    def nominal_flow_matrix(self) -> np.ndarray:
        return self._flow_matrix[0]

    @property
    def generator_map(self) -> np.ndarray:
        return self._generator_map

    def sample_response(
        self,
        dispatch: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw one uncertainty realization per dispatch and return violation margins.

        A value at most zero is secure.  Positive values give the worst fractional
        violation of a branch rating or generator bound.
        """

        p = np.atleast_2d(np.asarray(dispatch, dtype=float))
        if p.shape[1] != len(self.case.p_max):
            raise ValueError("dispatch has the wrong number of generators")
        n = len(p)
        common = rng.normal(0., .02, size=(n, 1))
        regional = rng.normal(0., .035, size=(n, 4))
        local = rng.normal(0., .02, size=(n, len(self._load_bus)))
        relative_error = common + regional[:, self._load_region] + local
        loads = np.broadcast_to(self.case.loads, (n, len(self.case.loads))).copy()
        loads[:, self._load_bus] *= np.clip(1. + relative_error, .75, 1.25)
        imbalance = loads.sum(axis=1) - self.case.demand
        actual_generation = p + imbalance[:, None] * self._participation
        injections = actual_generation @ self._generator_map.T - loads
        # Numerical roundoff in balancing is assigned to the reference bus.
        injections[:, 0] -= injections.sum(axis=1)
        contingency = np.zeros(n, dtype=int)
        is_outage = rng.random(n) < self.outage_probability
        contingency[is_outage] = rng.integers(
            1, self.contingencies + 1, size=np.count_nonzero(is_outage)
        )
        flows = np.einsum("nij,nj->ni", self._flow_matrix[contingency], injections)
        line_margin = np.max(np.abs(flows) / self.case.branch_rate - 1., axis=1)
        upper_margin = np.max(
            (actual_generation - self.case.p_max) / self.case.p_max, axis=1
        )
        lower_margin = np.max(
            (self.case.p_min - actual_generation) / self.case.p_max, axis=1
        )
        return np.maximum.reduce((line_margin, upper_margin, lower_margin))

    def _base_flow_matrix(self):
        laplacian = self._incidence.T @ (
            self._susceptance[:, None] * self._incidence
        )
        angle_map = np.zeros((len(self.case.loads), len(self.case.loads)))
        angle_map[1:, 1:] = np.linalg.inv(laplacian[1:, 1:])
        return self._susceptance[:, None] * self._incidence @ angle_map

    def _flow_matrices(self):
        matrices = []
        outage_lines = []
        n_bus = len(self.case.loads)
        for outage in range(len(self.case.branch_x)):
            active = np.ones(len(self.case.branch_x), dtype=bool)
            active[outage] = False
            incidence = self._incidence[active]
            weights = self._susceptance[active]
            laplacian = incidence.T @ (weights[:, None] * incidence)
            reduced = laplacian[1:, 1:]
            if np.linalg.matrix_rank(reduced, tol=1e-10) != n_bus - 1:
                continue
            inverse = np.linalg.inv(reduced)
            angle_map = np.zeros((n_bus, n_bus))
            angle_map[1:, 1:] = inverse
            flow = self._susceptance[:, None] * self._incidence @ angle_map
            flow[outage] = 0.
            matrices.append(flow)
            outage_lines.append(outage)
        if not matrices:
            raise ValueError("case has no connected single-line contingencies")
        return np.asarray(outage_lines, dtype=int), matrices
