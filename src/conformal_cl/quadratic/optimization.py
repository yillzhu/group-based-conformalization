"""Section 4.5.1: true, standard, CQR, and wrapper optimization formulations.

Historical implementation; known issues are deliberately retained.
See docs/findings.md and KNOWN-ISSUE markers.
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from gurobi_ml.keras.keras import add_keras_constr
import time


def optimize_compare(
    dim, main_model, quantile, wrapper, P, alpha, qhat, A_f, A_g, b_f, b_g, c_f, c_g
):
    model = gp.Model()
    model.Params.LogToConsole = 0
    model.Params.NonConvex = 2

    xx = model.addMVar(shape=dim, lb=-1, ub=1)

    # mean estimate and quantile estimate
    yy = model.addVar(lb=-GRB.INFINITY)
    yy_interval = model.addMVar(shape=2, lb=-GRB.INFINITY)
    yy_wrapper = model.addMVar(shape=(P, 1), lb=-GRB.INFINITY)
    wrapper_indicator = model.addMVar(shape=P, vtype=GRB.BINARY)

    # Objective function
    model.setObjective(xx @ A_f @ xx + b_f.T @ xx + c_f, GRB.MINIMIZE)

    # Feasibility constraint
    fc = model.addConstr(yy <= 0)

    # True optimal
    tc = model.addConstr(yy == xx @ A_g @ xx + b_g.T @ xx + c_g)

    start = time.time()
    model.optimize()
    end = time.time()
    t_true = end - start

    if model.status == GRB.OPTIMAL:
        opt_true = xx.X
    else:
        opt_true = np.nan

    # Embed learned main model
    model.remove(tc)
    kc_mean = add_keras_constr(model, main_model, xx, yy)

    start = time.time()
    model.optimize()
    end = time.time()
    t_abs = end - start

    if model.status == GRB.OPTIMAL:
        opt_abs = xx.X
    else:
        opt_abs = np.nan

    # Embed learned quantile models
    kc_mean.remove()
    kc_qu = add_keras_constr(model, quantile, xx, yy_interval)

    # Change constraints to prediction interval
    model.remove(fc)

    qc = model.addConstr(yy_interval[1] + qhat <= 0)

    start = time.time()
    model.optimize()
    end = time.time()
    t_cqr = end - start

    if model.status == GRB.OPTIMAL:
        opt_cqr = xx.X
    else:
        opt_cqr = np.nan

    # Model wrapper approach
    kc_qu.remove()
    model.remove(qc)

    for i in range(P):
        wc = add_keras_constr(model, wrapper[i], xx, yy_wrapper[i])
        model.addGenConstrIndicator(wrapper_indicator[i], True, yy_wrapper[i] <= 0)

    model.addConstr(wrapper_indicator.sum() >= P * (1 - alpha))

    start = time.time()
    model.optimize()
    end = time.time()
    t_wrapper = end - start

    if model.status == GRB.OPTIMAL:
        opt_wrapper = xx.X
    else:
        opt_wrapper = np.nan

    return opt_abs, opt_cqr, opt_wrapper, opt_true, t_abs, t_cqr, t_wrapper, t_true


def is_feasible(A_g, b_g, c_g, xhat):
    if np.any(np.isnan(xhat)):
        return -1
    x = xhat.reshape(-1, 1)
    return int((x.T @ A_g @ x + b_g.T @ x + c_g <= 0).item())
