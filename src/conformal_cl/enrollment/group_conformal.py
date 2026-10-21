"""Sections 4.4.3 and 4.6.2: per-applicant group-based enrollment adaptation.

Historical implementation; known issues are deliberately retained.
See docs/findings.md and KNOWN-ISSUE markers.
"""

from pathlib import Path
from conformal_cl.paths import PROJECT_ROOT
from conformal_cl.groups import Group
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from gurobi_ml import add_predictor_constr
from queue import Queue
import time
import logging
import torch
import pickle
import pandas as pd
from .models import create_quantile_net


def load_models_and_data():
    """
    Load trained quantile model, scaler, qhat, and calibration data.

    Returns:
        model_quantile: Trained quantile neural network
        scaler: Feature scaler for SAT and GPA
        qhat: Conformal adjustment value
        cal_X: Calibration features (scaled)
        cal_probs: Calibration probabilities (from basic model)
    """
    base_path = PROJECT_ROOT
    results_path = base_path / "results" / "enrollment"
    data_path = base_path / "data" / "college_student_enroll-s1-1.csv"

    # Load quantile model
    model_quantile = create_quantile_net(input_size=3, hidden_size=10, output_size=2)
    model_quantile.load_state_dict(
        torch.load(results_path / "enrollment_model_quantile.pth")
    )
    model_quantile.eval()

    # Load scaler
    with open(results_path / "enrollment_scaler_quantile.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Load qhat
    qhat_df = pd.read_csv(results_path / "enrollment_qhat.csv")
    qhat = qhat_df.loc[0, "qhat"]

    # Load calibration data
    df = pd.read_csv(data_path)
    features = df[["SAT", "GPA", "merit"]].values
    features[:, :2] = scaler.transform(features[:, :2])

    # For calibration, use 10% of data (last 2000 samples after 70% train, 20% test)
    # KNOWN-ISSUE[ISSUE-04]: last rows differ from randomized training calibration.
    cal_X = features[-2000:]

    # Load basic model to get calibration probabilities
    from .models import create_enrollment_net

    model_basic = create_enrollment_net(input_size=3, hidden_size=10, output_size=1)
    model_basic.load_state_dict(torch.load(results_path / "enrollment_model_basic.pth"))
    model_basic.eval()

    with torch.no_grad():
        cal_probs = model_basic(torch.FloatTensor(cal_X)).numpy()

    return model_quantile, scaler, qhat, cal_X, cal_probs


def run_enroll_rcc(
    sat_scaled,
    gpa_scaled,
    quantile_model,
    basic_model,
    cal_X,
    cal_probs,
    qhat_global,
    theta=0.6,
    alpha=0.1,
    delta=0.1,
    max_split=64,
    merit_lb=0.0,
    merit_ub=25.0,
):
    """
    Run RCC-based optimization for a single applicant.

    Args:
        sat_scaled: Scaled SAT score (fixed)
        gpa_scaled: Scaled GPA (fixed)
        quantile_model: Trained quantile neural network
        basic_model: Basic enrollment model for computing calibration probabilities
        cal_X: Calibration features [N, 3] (SAT, GPA, merit) - used to extract merit values
        cal_probs: Calibration probabilities from basic model [N, 1] - NOT USED (recomputed per applicant)
        qhat_global: Global conformal adjustment
        theta: Minimum enrollment probability threshold
        alpha: Miscoverage level for conformal prediction
        delta: Minimum group size parameter
        max_split: Maximum number of splits
        merit_lb: Lower bound for merit (thousands of dollars)
        merit_ub: Upper bound for merit (thousands of dollars)

    Returns:
        optimal_merit: Optimal merit amount
        is_feasible: Whether a feasible solution was found
        elapsed_time: Time taken
        num_splits: Number of splits performed
    """
    start_time = time.time()

    # Initialize: merit is 1D decision variable
    dim = 1

    # Extract merit values from calibration data (column 2)
    # For conformalization, we only need the decision variable (merit), not fixed features (SAT, GPA)
    cal_merit = cal_X[:, 2:3]  # Shape: (N, 1)

    # KNOWN-ISSUE[ISSUE-05]: constructed inputs/teacher targets; cal_probs and global qhat unused.
    # Create applicant-specific calibration data by pairing THIS applicant's (SAT, GPA)
    # with the calibration merit values
    n_cal = len(cal_merit)
    cal_X_applicant = np.zeros((n_cal, 3))
    cal_X_applicant[:, 0] = sat_scaled  # Fixed SAT for this applicant
    cal_X_applicant[:, 1] = gpa_scaled  # Fixed GPA for this applicant
    cal_X_applicant[:, 2] = cal_merit.flatten()  # Vary merit

    # Compute residuals for conformalization using THIS applicant's calibration set
    # For each calibration point, evaluate quantile model
    q_cal = np.zeros((n_cal, 2))
    with torch.no_grad():
        q_cal = quantile_model(torch.FloatTensor(cal_X_applicant)).numpy()

    # Compute calibration probabilities for THIS applicant using basic model
    with torch.no_grad():
        cal_probs_applicant = basic_model(torch.FloatTensor(cal_X_applicant)).numpy()

    # Nonconformity scores: max(y - upper, lower - y)
    R = np.maximum(
        cal_probs_applicant - q_cal[:, 1:2], q_cal[:, 0:1] - cal_probs_applicant
    )

    # Initialize queues
    groups_queue = Queue()
    groups_opt = Queue()  # Groups where lower_quantile + qhat >= theta (feasible)
    groups_nc = Queue()  # Groups without enough data for conformalization

    # Create initial group covering entire merit range [merit_lb, merit_ub]
    G = Group(bounds={})
    G.conformalize(cal_merit, R, alpha=alpha, delta=delta)
    groups_queue.put(G)

    # Create Gurobi model for finding extrema
    model = gp.Model("enroll_rcc_extrema")
    model.Params.LogToConsole = 0

    # Decision variable: merit (1D)
    merit_var = model.addVar(lb=merit_lb, ub=merit_ub, name="merit")

    # Input to neural network: [sat_scaled (fixed), gpa_scaled (fixed), merit (variable)]
    # We'll need to create fixed variables for SAT and GPA
    sat_var = model.addVar(lb=sat_scaled, ub=sat_scaled, name="sat")
    gpa_var = model.addVar(lb=gpa_scaled, ub=gpa_scaled, name="gpa")

    # Output from quantile model: [lower_quantile, upper_quantile]
    quantile_output = model.addMVar(shape=2, lb=-GRB.INFINITY, name="quantiles")

    # Embed quantile model
    input_vars = [sat_var, gpa_var, merit_var]

    num_splits = 0

    # KNOWN-ISSUE[ISSUE-06]: split limit leaves candidates out of final optimization.
    while not groups_queue.empty() and num_splits < max_split:
        # Remove all constraints (keep variable bounds)
        # KNOWN-ISSUE[ISSUE-09]: predictor lifecycle requires review; unchanged.
        model.remove(model.getConstrs())

        # Re-embed the quantile model (needed after removing constraints)
        add_predictor_constr(model, quantile_model, input_vars, quantile_output)

        G = groups_queue.get()

        # Add group boundary constraints
        for coeffs, d in G.bounds.items():
            coeffs = np.array(coeffs)
            norm = coeffs[:-1]  # Coefficients for decision variables
            b = coeffs[-1]  # RHS
            # d=1 means norm @ x <= b; d=-1 means norm @ x >= b
            model.addConstr(d * (norm[0] * merit_var) <= d * b)

        # STEP 1: Find MAXIMUM of (theta - lower_quantile - qhat)
        # In RCC.py this was: MAXIMIZE (upper_quantile + qhat)
        # For enrollment: constraint is lower_quantile + qhat >= theta
        # So we check: MAXIMIZE (theta - lower_quantile - qhat)
        # If this is <= 0, the constraint is satisfied everywhere (entire group is feasible)
        if np.any(np.isnan(G.max_x)):
            # KNOWN-ISSUE[ISSUE-02, ISSUE-03]: historical sign and cache convention retained.
            model.setObjective(theta - quantile_output[0] - G.qhat, GRB.MAXIMIZE)
            model.optimize()

            if model.status == GRB.OPTIMAL:
                G.max_x = np.array([merit_var.X])
                G.max_v = model.ObjVal
            else:
                continue

        # If max(theta - lower_quantile - qhat) <= 0, entire group is feasible
        # This means: lower_quantile + qhat >= theta everywhere
        if G.max_v <= 0.01:
            groups_opt.put(G)
            continue

        # STEP 2: Find MINIMUM of (theta - upper_quantile - qhat)
        # In RCC.py this was: MINIMIZE (lower_quantile - qhat)
        # For enrollment: if upper_quantile + qhat < theta, constraint is violated
        # So we check: MINIMIZE (theta - upper_quantile - qhat)
        # If this is >= 0, the constraint is violated everywhere (entire group is infeasible)
        if np.any(np.isnan(G.min_x)):
            model.setObjective(theta - quantile_output[1] - G.qhat, GRB.MINIMIZE)
            model.optimize()

            if model.status == GRB.OPTIMAL:
                G.min_x = np.array([merit_var.X])
                G.min_v = model.ObjVal
            else:
                continue

        # If min(theta - upper_quantile - qhat) >= 0, entire group is infeasible
        # This means: upper_quantile + qhat < theta everywhere
        if G.min_v >= -0.01:
            continue

        # STEP 3: Find MINIMUM of (theta - lower_quantile - qhat)
        # In RCC.py this was: MINIMIZE (upper_quantile + qhat)
        # This helps us find the split point
        if np.any(np.isnan(G.ul_x)):
            # KNOWN-ISSUE[ISSUE-02, ISSUE-03]: historical sign and cache convention retained.
            model.setObjective(theta - quantile_output[0] - G.qhat, GRB.MINIMIZE)
            model.optimize()

            if model.status == GRB.OPTIMAL:
                G.ul_x = np.array([merit_var.X])
                G.ul_v = model.ObjVal
            else:
                continue

        # Need to split this group
        # Find split point using linear interpolation
        if G.ul_v <= -0.01:
            # Interpolate between ul_x and max_x
            bound_split = G.ul_x - G.ul_v * (G.max_x - G.ul_x) / (G.max_v - G.ul_v)
        else:
            # Simple midpoint split
            bound_split = (G.ul_x + G.max_x) / 2

        # Create splitting hyperplane perpendicular to (max_x - ul_x)
        coeff_split = np.zeros(dim + 1)
        norm_split = G.max_x - G.ul_x
        b_split = np.dot(norm_split, bound_split)
        coeff_split[:dim] = norm_split
        coeff_split[dim] = b_split

        # Avoid splitting on the same hyperplane twice
        if tuple(coeff_split) not in G.bounds:
            G_l, G_r = G.split(coeff_split, cal_merit, R, alpha=alpha, delta=delta)

            # Only add groups with sufficient calibration data
            if G_l.Nhat >= 10 and G_l.Xhat == 1:
                groups_queue.put(G_l)
            else:
                groups_nc.put(G_l)

            if G_r.Nhat >= 10 and G_r.Xhat == 1:
                groups_queue.put(G_r)
            else:
                groups_nc.put(G_r)

            num_splits += 1

    # Now optimize over feasible groups
    if groups_opt.empty() and groups_nc.empty():
        elapsed_time = time.time() - start_time
        return None, False, elapsed_time, num_splits

    # Create optimization model
    opt_model = gp.Model("enroll_optimization")
    opt_model.Params.LogToConsole = 0

    merit_opt = opt_model.addVar(lb=merit_lb, ub=merit_ub, name="merit")
    sat_opt = opt_model.addVar(lb=sat_scaled, ub=sat_scaled, name="sat")
    gpa_opt = opt_model.addVar(lb=gpa_scaled, ub=gpa_scaled, name="gpa")

    # Objective: minimize merit
    opt_model.setObjective(merit_opt, GRB.MINIMIZE)

    optimal_merit = np.inf
    best_solution_found = False

    # Optimize over groups with guaranteed feasibility (no quantile constraint needed)
    while not groups_opt.empty():
        G_opt = groups_opt.get()
        opt_model.remove(opt_model.getConstrs())

        # Add group boundary constraints
        for coeffs, d in G_opt.bounds.items():
            coeffs = np.array(coeffs)
            norm = coeffs[:-1]
            b = coeffs[-1]
            opt_model.addConstr(d * (norm[0] * merit_opt) <= d * b)

        opt_model.optimize()

        if opt_model.status == GRB.OPTIMAL:
            if opt_model.ObjVal < optimal_merit:
                optimal_merit = opt_model.ObjVal
                best_solution_found = True

    # Optimize over groups without sufficient conformalization (add quantile constraint)
    if not groups_nc.empty():
        quantile_opt = opt_model.addMVar(shape=2, lb=-GRB.INFINITY, name="quantiles")
        input_vars_opt = [sat_opt, gpa_opt, merit_opt]

        while not groups_nc.empty():
            G_nc = groups_nc.get()
            opt_model.remove(opt_model.getConstrs())

            # Add quantile model constraint
            add_predictor_constr(
                opt_model, quantile_model, input_vars_opt, quantile_opt
            )
            # KNOWN-ISSUE[ISSUE-02, ISSUE-06]: plus sign and zero fallback retained.
            opt_model.addConstr(quantile_opt[0] + G_nc.qhat >= theta)

            # Add group boundary constraints
            for coeffs, d in G_nc.bounds.items():
                coeffs = np.array(coeffs)
                norm = coeffs[:-1]
                b = coeffs[-1]
                opt_model.addConstr(d * (norm[0] * merit_opt) <= d * b)

            opt_model.optimize()

            if opt_model.status == GRB.OPTIMAL:
                if opt_model.ObjVal < optimal_merit:
                    optimal_merit = opt_model.ObjVal
                    best_solution_found = True

    elapsed_time = time.time() - start_time

    if best_solution_found:
        return optimal_merit, True, elapsed_time, num_splits
    else:
        return None, False, elapsed_time, num_splits


def run_enroll_rcc_batch(
    applicants_df,
    features_scaled,
    quantile_model,
    basic_model,
    cal_X,
    cal_probs,
    qhat_global,
    theta=0.6,
    alpha=0.1,
    delta=0.1,
    max_split=64,
    merit_lb=0.0,
    merit_ub=25.0,
):
    """
    Run RCC optimization for multiple applicants.

    Args:
        applicants_df: DataFrame with applicant data
        features_scaled: Scaled features [N, 2] (SAT, GPA)
        quantile_model: Trained quantile neural network
        basic_model: Basic enrollment model for calibration probabilities
        cal_X: Calibration features
        cal_probs: Calibration probabilities
        qhat_global: Global conformal adjustment
        theta: Minimum enrollment probability
        alpha: Miscoverage level
        delta: Minimum group size parameter
        max_split: Maximum number of splits per applicant
        merit_lb: Lower bound for merit
        merit_ub: Upper bound for merit

    Returns:
        results_df: DataFrame with optimization results
    """
    results = []

    for i in range(len(applicants_df)):
        student_id = applicants_df.loc[i, "StudentID"]
        sat_scaled = features_scaled[i, 0]
        gpa_scaled = features_scaled[i, 1]

        optimal_merit, is_feasible, elapsed_time, num_splits = run_enroll_rcc(
            sat_scaled=sat_scaled,
            gpa_scaled=gpa_scaled,
            quantile_model=quantile_model,
            basic_model=basic_model,
            cal_X=cal_X,
            cal_probs=cal_probs,
            qhat_global=qhat_global,
            theta=theta,
            alpha=alpha,
            delta=delta,
            max_split=max_split,
            merit_lb=merit_lb,
            merit_ub=merit_ub,
        )

        results.append(
            {
                "StudentID": student_id,
                "SAT": applicants_df.loc[i, "SAT"],
                "GPA": applicants_df.loc[i, "GPA"],
                "OptimalMerit": optimal_merit,
                "IsFeasible": is_feasible,
                "NumSplits": num_splits,
                "Time": elapsed_time,
                "Theta": theta,
            }
        )

        if (i + 1) % 100 == 0:
            logging.info(f"Processed {i + 1}/{len(applicants_df)} applicants")

    return pd.DataFrame(results)
