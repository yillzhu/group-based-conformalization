"""Sections 4.4.3 and 4.5: Algorithm 1 applied to quadratic constraints."""

from conformal_cl import groups as group
from . import training
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from gurobi_ml.keras.keras import add_keras_constr
from queue import Queue
from conformal_cl.solver import quiet_gurobi_env


def run_rcc(
    samples, alpha, delta, max_split=64, quantile=None, feasibility_tol=1e-6
):
    (X, y, cal_X, cal_y, A_f, A_g, b_f, b_g, c_f, c_g) = samples

    dim = np.shape(A_f)[0]
    if quantile is None:
        quantile = training.train_quantiles(dim, X, y, alpha)

    q = quantile.predict(cal_X, verbose=0)
    R = np.maximum(cal_y - q[:, 1:2], q[:, 0:1] - cal_y)

    solve_time = 0.0
    groups_queue = Queue()
    groups_opt = Queue()
    groups_nc = Queue()
    G = group.Group(bounds={})
    G.conformalize(cal_X, R, alpha=alpha, delta=delta)
    if G.Xhat:
        groups_queue.put(G)
    else:
        groups_nc.put(G)

    count = 0
    while not groups_queue.empty():
        G = groups_queue.get()

        # A Keras embedding adds auxiliary/general constraints.  Use a fresh model
        # per region so no embedding state leaks across partitions.
        model = gp.Model(env=quiet_gurobi_env())
        xx = model.addMVar(shape=dim, lb=-1, ub=1)
        yy_interval = model.addMVar(shape=2, lb=-GRB.INFINITY)
        add_keras_constr(model, quantile, xx, yy_interval)

        for coeffs, d in G.bounds.items():
            coeffs = np.array(coeffs)
            norm = coeffs[:-1]
            b = coeffs[-1]
            model.addConstr(d * (norm @ xx) <= d * b)

        if np.any(np.isnan(G.max_x)):
            # Objective function
            model.setObjective(yy_interval[1] + G.qhat, GRB.MAXIMIZE)

            model.optimize()
            solve_time += model.Runtime

            if model.status == GRB.OPTIMAL:
                G.max_x = xx.X
                G.max_v = model.ObjVal
            else:
                groups_nc.put(G)
                continue

        if G.max_v <= feasibility_tol:
            groups_opt.put(G)
            continue

        if np.any(np.isnan(G.min_x)):
            # Objective function
            model.setObjective(yy_interval[0] - G.qhat, GRB.MINIMIZE)

            model.optimize()
            solve_time += model.Runtime

            if model.status == GRB.OPTIMAL:
                G.min_x = xx.X
                G.min_v = model.ObjVal
            else:
                groups_nc.put(G)
                continue

        if G.min_v >= -feasibility_tol:
            continue

        # if np.linalg.norm(G.min_x - G.max_x) <= 0.01 * np.sqrt(dim):
        #    continue

        if np.any(np.isnan(G.ul_x)):
            # Objective function
            model.setObjective(yy_interval[1] + G.qhat, GRB.MINIMIZE)

            model.optimize()
            solve_time += model.Runtime

            if model.status == GRB.OPTIMAL:
                G.ul_x = xx.X
                G.ul_v = model.ObjVal
            else:
                groups_nc.put(G)
                continue

        if G.ul_v <= -feasibility_tol and not np.isclose(
            G.max_v, G.ul_v, atol=feasibility_tol, rtol=0
        ):
            bound_split = G.ul_x - G.ul_v * (G.max_x - G.ul_x) / (G.max_v - G.ul_v)
        else:
            bound_split = (G.ul_x + G.max_x) / 2

        coeff_split = np.zeros(dim + 1)
        norm_split = G.max_x - G.ul_x
        norm_length = np.linalg.norm(norm_split)
        if not np.isfinite(norm_length) or norm_length <= feasibility_tol:
            groups_nc.put(G)
            continue
        # A hyperplane is invariant to positive scaling. Unit normalization keeps
        # repeated regional membership calculations well conditioned.
        norm_split = norm_split / norm_length
        b_split = np.dot(norm_split, bound_split)
        coeff_split[:dim] = norm_split
        coeff_split[dim] = b_split

        if tuple(coeff_split) not in G.bounds:
            G_l, G_r = G.split(
                coeff_split, cal_X, R, alpha=alpha, delta=delta
            )

            if G_l.Xhat:
                groups_queue.put(G_l)
            else:
                groups_nc.put(G_l)

            if G_r.Xhat:
                groups_queue.put(G_r)
            else:
                groups_nc.put(G_r)
        else:
            groups_nc.put(G)

        count += 1
        if count == max_split:
            break
        # print(coeff_split)
        # print(G.min_v, G.max_v, G.min_x, G.max_x)

    # A split cap is not evidence of infeasibility.  Evaluate every unresolved
    # region using its inherited/local conformal correction.
    while not groups_queue.empty():
        groups_nc.put(groups_queue.get())

    # calculate the feasibility of true model
    model3 = gp.Model(env=quiet_gurobi_env())
    xx3 = model3.addMVar(shape=dim, lb=-1, ub=1)
    model3.setObjective(xx3 @ A_g @ xx3 + b_g.T @ xx3 + c_g, GRB.MINIMIZE)
    model3.optimize()
    if model3.status == GRB.OPTIMAL:
        true_obj = model3.ObjVal
        true_f = true_obj <= 0
    else:
        true_obj = np.nan
        true_f = np.nan

    if groups_opt.empty() and groups_nc.empty():
        return np.nan, [2], solve_time, true_f, true_obj

    obj = np.inf
    while not groups_opt.empty():
        G_opt = groups_opt.get()
        model2 = gp.Model(env=quiet_gurobi_env())
        xx2 = model2.addMVar(shape=dim, lb=-1, ub=1)
        model2.setObjective(xx2 @ A_f @ xx2 + b_f.T @ xx2 + c_f, GRB.MINIMIZE)
        for coeffs, d in G_opt.bounds.items():
            coeffs = np.array(coeffs)
            norm = coeffs[:-1]
            b = coeffs[-1]
            model2.addConstr(d * (norm @ xx2) <= d * b)

        model2.optimize()
        solve_time += model2.Runtime

        if model2.status == GRB.OPTIMAL:
            if model2.ObjVal < obj:
                opt = xx2.X
                obj = model2.ObjVal

    while not groups_nc.empty():
        G_nc = groups_nc.get()
        model2 = gp.Model(env=quiet_gurobi_env())
        xx2 = model2.addMVar(shape=dim, lb=-1, ub=1)
        yy2_interval = model2.addMVar(shape=2, lb=-GRB.INFINITY)
        model2.setObjective(xx2 @ A_f @ xx2 + b_f.T @ xx2 + c_f, GRB.MINIMIZE)
        add_keras_constr(model2, quantile, xx2, yy2_interval)
        model2.addConstr(yy2_interval[1] + G_nc.qhat <= 0)
        for coeffs, d in G_nc.bounds.items():
            coeffs = np.array(coeffs)
            norm = coeffs[:-1]
            b = coeffs[-1]
            model2.addConstr(d * (norm @ xx2) <= d * b)

        model2.optimize()
        solve_time += model2.Runtime

        if model2.status == GRB.OPTIMAL:
            if model2.ObjVal < obj:
                opt = xx2.X
                obj = model2.ObjVal

    if not np.isfinite(obj):
        return np.nan, [2], solve_time, true_f, true_obj

    is_f = opt.T @ A_g @ opt + b_g.T @ opt + c_g <= 0

    return opt, is_f[0], solve_time, true_f, true_obj
