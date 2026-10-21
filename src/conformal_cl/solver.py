"""Shared Gurobi environment helpers."""

from __future__ import annotations

import atexit

import gurobipy as gp


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


def _dispose_quiet_env():
    global _QUIET_ENV
    if _QUIET_ENV is not None:
        _QUIET_ENV.dispose()
        _QUIET_ENV = None


atexit.register(_dispose_quiet_env)
