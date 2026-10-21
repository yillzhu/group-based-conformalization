"""Sections 4.4.3 and 4.5: Algorithm 1 applied to quadratic constraints.

Historical implementation; known issues are deliberately retained.
See docs/findings.md and KNOWN-ISSUE markers.
"""

from conformal_cl import groups as group
from . import training
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from gurobi_ml.keras.keras import add_keras_constr
from queue import Queue
import time


def run_rcc(samples, alpha, delta, max_split=64):
    (X, y, cal_X, cal_y, A_f, A_g, b_f, b_g, c_f, c_g) = samples

    dim = np.shape(A_f)[0]
    quantile = training.train_quantiles(dim, X, y, alpha)

    q = quantile.predict(cal_X, verbose=0)
    R = np.maximum(cal_y - q[:, 1:2], q[:, 0:1] - cal_y)

    # KNOWN-ISSUE[ISSUE-08]: RCC timer includes construction, partitioning, and extra solves.
    start = time.time()
    groups_queue = Queue()
    groups_opt = Queue()
    groups_nc = Queue()
    G = group.Group(bounds={})
    # KNOWN-ISSUE[ISSUE-07]: supplied alpha/delta not forwarded.
    G.conformalize(cal_X, R)
    groups_queue.put(G)

    model = gp.Model()
    # model.Params.LogToConsole = 0

    xx = model.addMVar(shape=dim, lb=-1, ub=1)
    yy_interval = model.addMVar(shape=2, lb=-GRB.INFINITY)

    count = 0
    while not groups_queue.empty():
        # KNOWN-ISSUE[ISSUE-09]: predictor cleanup/re-embedding requires solver review.
        model.remove(model.getConstrs())
        # Embed learned quantile models
        kc_qu = add_keras_constr(model, quantile, xx, yy_interval)

        G = groups_queue.get()

        for coeffs, d in G.bounds.items():
            coeffs = np.array(coeffs)
            norm = coeffs[:-1]
            b = coeffs[-1]
            model.addConstr(d * (norm @ xx) <= d * b)

        if np.any(np.isnan(G.max_x)):
            # Objective function
            model.setObjective(yy_interval[1] + G.qhat, GRB.MAXIMIZE)

            model.optimize()

            if model.status == GRB.OPTIMAL:
                G.max_x = xx.X
                G.max_v = model.ObjVal

        if G.max_v <= 0.01:
            groups_opt.put(G)
            continue

        if np.any(np.isnan(G.min_x)):
            # Objective function
            model.setObjective(yy_interval[0] - G.qhat, GRB.MINIMIZE)

            model.optimize()

            if model.status == GRB.OPTIMAL:
                G.min_x = xx.X
                G.min_v = model.ObjVal

        if G.min_v >= -0.01:
            continue

        # if np.linalg.norm(G.min_x - G.max_x) <= 0.01 * np.sqrt(dim):
        #    continue

        if np.any(np.isnan(G.ul_x)):
            # Objective function
            model.setObjective(yy_interval[1] + G.qhat, GRB.MINIMIZE)

            model.optimize()

            if model.status == GRB.OPTIMAL:
                G.ul_x = xx.X
                G.ul_v = model.ObjVal

        if G.ul_v <= -0.01:
            bound_split = G.ul_x - G.ul_v * (G.max_x - G.ul_x) / (G.max_v - G.ul_v)
        else:
            bound_split = (G.ul_x + G.max_x) / 2

        coeff_split = np.zeros(dim + 1)
        norm_split = G.max_x - G.ul_x
        b_split = np.dot(G.max_x - G.ul_x, bound_split)
        coeff_split[:dim] = norm_split
        coeff_split[dim:] = b_split

        if tuple(coeff_split) not in G.bounds:
            # KNOWN-ISSUE[ISSUE-07]: split also uses default alpha/delta.
            G_l, G_r = G.split(coeff_split, cal_X, R)

            if G_l.Nhat >= 10 and G_l.Xhat == 1:
                groups_queue.put(G_l)
            else:
                groups_nc.put(G_l)

            if G_r.Nhat >= 10 and G_r.Xhat == 1:
                groups_queue.put(G_r)
            else:
                groups_nc.put(G_r)

        count += 1
        # KNOWN-ISSUE[ISSUE-06]: pending candidates are not moved to final queues.
        if count == max_split:
            break
        # print(coeff_split)
        # print(G.min_v, G.max_v, G.min_x, G.max_x)

    # calculate the feasibility of true model
    model3 = gp.Model()
    model3.Params.LogToConsole = 0
    xx3 = model3.addMVar(shape=dim, lb=-1, ub=1)
    model3.setObjective(xx3 @ A_g @ xx3 + b_g.T @ xx3 + c_g, GRB.MINIMIZE)
    model3.optimize()
    true_f = model3.ObjVal <= 0

    if groups_opt.empty() and groups_nc.empty():
        end = time.time()
        return np.nan, [2], end - start, true_f, model3.ObjVal

    model2 = gp.Model()
    model2.Params.LogToConsole = 0
    xx2 = model2.addMVar(shape=dim, lb=-1, ub=1)
    model2.setObjective(xx2 @ A_f @ xx2 + b_f.T @ xx2 + c_f, GRB.MINIMIZE)

    obj = np.inf
    while not groups_opt.empty():
        G_opt = groups_opt.get()
        model2.remove(model2.getConstrs())
        for coeffs, d in G_opt.bounds.items():
            coeffs = np.array(coeffs)
            norm = coeffs[:-1]
            b = coeffs[-1]
            model2.addConstr(d * (norm @ xx2) <= d * b)

        model2.optimize()

        if model2.status == GRB.OPTIMAL:
            if model2.ObjVal < obj:
                opt = xx2.X
                obj = model2.ObjVal

    yy2_interval = model2.addMVar(shape=2, lb=-GRB.INFINITY)
    while not groups_nc.empty():
        G_nc = groups_nc.get()
        model2.remove(model2.getConstrs())
        add_keras_constr(model2, quantile, xx2, yy2_interval)
        model2.addConstr(yy2_interval[1] + G_nc.qhat <= 0)
        for coeffs, d in G_nc.bounds.items():
            coeffs = np.array(coeffs)
            norm = coeffs[:-1]
            b = coeffs[-1]
            model2.addConstr(d * (norm @ xx2) <= d * b)

        model2.optimize()

        if model2.status == GRB.OPTIMAL:
            if model2.ObjVal < obj:
                opt = xx2.X
                obj = model2.ObjVal

    if obj > 100:
        end = time.time()
        return np.nan, [2], end - start, true_f, model3.ObjVal

    end = time.time()

    is_f = opt.T @ A_g @ opt + b_g.T @ opt + c_g <= 0

    return opt, is_f[0], end - start, true_f, model3.ObjVal
