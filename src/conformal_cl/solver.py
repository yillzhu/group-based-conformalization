"""Shared Gurobi environment helpers."""

from __future__ import annotations

import atexit

import gurobipy as gp
from gurobipy import GRB


_QUIET_ENV = None


def quiet_gurobi_env():
    """Return one lazily initialized, process-local quiet Gurobi environment."""

    global _QUIET_ENV
    if _QUIET_ENV is None:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        _QUIET_ENV = env
    return _QUIET_ENV


def gurobi_status_label(model_or_status):
    """Return a stable, human-readable label for a Gurobi status code."""

    status = (
        model_or_status.status
        if hasattr(model_or_status, "status")
        else int(model_or_status)
    )
    labels = {
        GRB.LOADED: "Loaded",
        GRB.OPTIMAL: "Optimal",
        GRB.INFEASIBLE: "Infeasible",
        GRB.INF_OR_UNBD: "InfeasibleOrUnbounded",
        GRB.UNBOUNDED: "Unbounded",
        GRB.CUTOFF: "Cutoff",
        GRB.ITERATION_LIMIT: "IterationLimit",
        GRB.NODE_LIMIT: "NodeLimit",
        GRB.TIME_LIMIT: "TimeLimit",
        GRB.SOLUTION_LIMIT: "SolutionLimit",
        GRB.INTERRUPTED: "Interrupted",
        GRB.NUMERIC: "NumericalFailure",
        GRB.SUBOPTIMAL: "Suboptimal",
        GRB.USER_OBJ_LIMIT: "UserObjectiveLimit",
        GRB.WORK_LIMIT: "WorkLimit",
        GRB.MEM_LIMIT: "MemoryLimit",
    }
    return labels.get(status, f"SolverStatus{status}")


def _dispose_quiet_env():
    global _QUIET_ENV
    if _QUIET_ENV is not None:
        _QUIET_ENV.dispose()
        _QUIET_ENV = None


atexit.register(_dispose_quiet_env)
