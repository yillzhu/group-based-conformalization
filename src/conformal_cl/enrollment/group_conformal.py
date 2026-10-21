"""Sections 4.4.3 and 4.6.2: per-applicant group-based enrollment adaptation."""

from pathlib import Path
from conformal_cl.paths import PROJECT_ROOT
from conformal_cl.groups import Group
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from gurobi_ml import add_predictor_constr
from queue import Queue
import logging
import torch
import pickle
import pandas as pd
from .models import create_quantile_net
from .data import split_train_test_cal_indices
from conformal_cl.solver import quiet_gurobi_env


def load_models_and_data(results_path=None):
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
    if results_path is None:
        results_path = base_path / "results" / "corrected" / "enrollment"
    results_path = Path(results_path)
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

    split_ids_path = results_path / "enrollment_split_ids.csv"
    if split_ids_path.exists():
        split_ids = pd.read_csv(split_ids_path)
        calibration_ids = set(
            split_ids.loc[split_ids["Split"].eq("calibration"), "StudentID"]
        )
        cal_X = features[df["StudentID"].isin(calibration_ids).to_numpy()]
    else:
        _, _, calibration_idx = split_train_test_cal_indices(len(df))
        cal_X = features[calibration_idx]

    # Load basic model to get calibration probabilities
    from .models import create_enrollment_net

    model_basic = create_enrollment_net(input_size=3, hidden_size=10, output_size=1)
    model_basic.load_state_dict(torch.load(results_path / "enrollment_model_basic.pth"))
    model_basic.eval()

    with torch.no_grad():
        cal_probs = model_basic(torch.FloatTensor(cal_X)).numpy()

    return model_quantile, model_basic, scaler, qhat, cal_X, cal_probs


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
    feasibility_tol=1e-6,
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
        solve_time: Sum of Gurobi-reported solve runtimes
        num_splits: Number of splits performed
        status: Stable outcome label
        diagnostics: Selected route and local conformal correction
    """
    solve_time = 0.0

    # Initialize: merit is 1D decision variable
    dim = 1

    # Extract merit values from calibration data (column 2)
    # For conformalization, we only need the decision variable (merit), not fixed features (SAT, GPA)
    cal_merit = cal_X[:, 2:3]  # Shape: (N, 1)

    # The current method constructs applicant-specific teacher targets.  This is
    # not evidence of coverage for latent true enrollment probabilities; callers
    # and reports must describe the target accordingly.
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
    groups_opt = Queue()  # Groups where lower_quantile - qhat >= theta (feasible)
    groups_nc = Queue()  # Groups without enough data for conformalization

    # Create initial group covering entire merit range [merit_lb, merit_ub]
    G = Group(bounds={}, qhat=qhat_global)
    G.conformalize(cal_merit, R, alpha=alpha, delta=delta)
    if G.Xhat:
        groups_queue.put(G)
    else:
        groups_nc.put(G)

    num_splits = 0

    while not groups_queue.empty() and num_splits < max_split:
        G = groups_queue.get()

        # Predictor embeddings create auxiliary variables and general constraints;
        # build a fresh model per region instead of trying to partially remove a
        # previous embedding.
        model = gp.Model("enroll_rcc_extrema", env=quiet_gurobi_env())
        merit_var = model.addVar(lb=merit_lb, ub=merit_ub, name="merit")
        sat_var = model.addVar(lb=sat_scaled, ub=sat_scaled, name="sat")
        gpa_var = model.addVar(lb=gpa_scaled, ub=gpa_scaled, name="gpa")
        quantile_output = model.addMVar(
            shape=2, lb=-GRB.INFINITY, name="quantiles"
        )
        add_predictor_constr(
            model,
            quantile_model,
            [sat_var, gpa_var, merit_var],
            quantile_output,
        )

        # Add group boundary constraints
        for coeffs, d in G.bounds.items():
            coeffs = np.array(coeffs)
            norm = coeffs[:-1]  # Coefficients for decision variables
            b = coeffs[-1]  # RHS
            # d=1 means norm @ x <= b; d=-1 means norm @ x >= b
            model.addConstr(d * (norm[0] * merit_var) <= d * b)

        # STEP 1: Find MAXIMUM of (theta - lower_quantile + qhat)
        # In RCC.py this was: MAXIMIZE (upper_quantile + qhat)
        # The calibrated lower endpoint is lower_quantile - qhat.
        # So we check: MAXIMIZE (theta - lower_quantile + qhat)
        # If this is <= 0, the constraint is satisfied everywhere (entire group is feasible)
        if np.any(np.isnan(G.max_x)):
            model.setObjective(theta - quantile_output[0] + G.qhat, GRB.MAXIMIZE)
            model.optimize()
            solve_time += model.Runtime

            if model.status == GRB.OPTIMAL:
                G.max_x = np.array([merit_var.X])
                G.max_v = model.ObjVal
            else:
                groups_nc.put(G)
                continue

        # If the maximum violation is nonpositive (up to solver tolerance), the
        # entire group is feasible.
        # This means: lower_quantile - qhat >= theta everywhere
        if G.max_v <= feasibility_tol:
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
            solve_time += model.Runtime

            if model.status == GRB.OPTIMAL:
                G.min_x = np.array([merit_var.X])
                G.min_v = model.ObjVal
            else:
                groups_nc.put(G)
                continue

        # If min(theta - upper_quantile - qhat) >= 0, entire group is infeasible
        # This means: upper_quantile + qhat < theta everywhere
        if G.min_v >= -feasibility_tol:
            continue

        # STEP 3: Find MINIMUM of (theta - lower_quantile + qhat)
        # In RCC.py this was: MINIMIZE (upper_quantile + qhat)
        # This helps us find the split point
        if np.any(np.isnan(G.ul_x)):
            model.setObjective(theta - quantile_output[0] + G.qhat, GRB.MINIMIZE)
            model.optimize()
            solve_time += model.Runtime

            if model.status == GRB.OPTIMAL:
                G.ul_x = np.array([merit_var.X])
                G.ul_v = model.ObjVal
            else:
                groups_nc.put(G)
                continue

        # Need to split this group
        # Find split point using linear interpolation
        if G.ul_v <= -feasibility_tol and not np.isclose(
            G.max_v, G.ul_v, atol=feasibility_tol, rtol=0
        ):
            # Interpolate between ul_x and max_x
            bound_split = G.ul_x - G.ul_v * (G.max_x - G.ul_x) / (G.max_v - G.ul_v)
        else:
            # Simple midpoint split
            bound_split = (G.ul_x + G.max_x) / 2

        # Create splitting hyperplane perpendicular to (max_x - ul_x)
        coeff_split = np.zeros(dim + 1)
        norm_split = G.max_x - G.ul_x
        norm_length = np.linalg.norm(norm_split)
        if not np.isfinite(norm_length) or norm_length <= feasibility_tol:
            groups_nc.put(G)
            continue
        norm_split = norm_split / norm_length
        b_split = np.dot(norm_split, bound_split)
        coeff_split[:dim] = norm_split
        coeff_split[dim] = b_split

        # Avoid splitting on the same hyperplane twice
        if tuple(coeff_split) not in G.bounds:
            G_l, G_r = G.split(coeff_split, cal_merit, R, alpha=alpha, delta=delta)

            # Only add groups with sufficient calibration data
            if G_l.Xhat:
                groups_queue.put(G_l)
            else:
                groups_nc.put(G_l)

            if G_r.Xhat:
                groups_queue.put(G_r)
            else:
                groups_nc.put(G_r)

            num_splits += 1
        else:
            groups_nc.put(G)

    # A split cap is a computational terminal condition, not a proof that the
    # remaining regions are infeasible.  Route every pending group through the
    # explicit quantile-constrained fallback.
    while not groups_queue.empty():
        groups_nc.put(groups_queue.get())

    # Now optimize over feasible and unresolved groups.
    if groups_opt.empty() and groups_nc.empty():
        return (
            None,
            False,
            solve_time,
            num_splits,
            "NoCertifiedSolution",
            {"route": None, "qhat": None},
        )

    optimal_merit = np.inf
    best_solution_found = False
    best_diagnostics = {"route": None, "qhat": None}

    # Optimize over groups with guaranteed feasibility (no quantile constraint needed)
    while not groups_opt.empty():
        G_opt = groups_opt.get()
        opt_model = gp.Model(
            "enroll_optimization_certified", env=quiet_gurobi_env()
        )
        merit_opt = opt_model.addVar(lb=merit_lb, ub=merit_ub, name="merit")
        opt_model.setObjective(merit_opt, GRB.MINIMIZE)

        # Add group boundary constraints
        for coeffs, d in G_opt.bounds.items():
            coeffs = np.array(coeffs)
            norm = coeffs[:-1]
            b = coeffs[-1]
            opt_model.addConstr(d * (norm[0] * merit_opt) <= d * b)

        opt_model.optimize()
        solve_time += opt_model.Runtime

        if opt_model.status == GRB.OPTIMAL:
            if opt_model.ObjVal < optimal_merit:
                optimal_merit = opt_model.ObjVal
                best_solution_found = True
                best_diagnostics = {
                    "route": "certified_region",
                    "qhat": float(G_opt.qhat),
                }

    # Optimize over groups without sufficient conformalization (add quantile constraint)
    while not groups_nc.empty():
        G_nc = groups_nc.get()
        opt_model = gp.Model(
            "enroll_optimization_unresolved", env=quiet_gurobi_env()
        )
        merit_opt = opt_model.addVar(lb=merit_lb, ub=merit_ub, name="merit")
        sat_opt = opt_model.addVar(lb=sat_scaled, ub=sat_scaled, name="sat")
        gpa_opt = opt_model.addVar(lb=gpa_scaled, ub=gpa_scaled, name="gpa")
        quantile_opt = opt_model.addMVar(
            shape=2, lb=-GRB.INFINITY, name="quantiles"
        )
        opt_model.setObjective(merit_opt, GRB.MINIMIZE)
        add_predictor_constr(
            opt_model,
            quantile_model,
            [sat_opt, gpa_opt, merit_opt],
            quantile_opt,
        )
        opt_model.addConstr(quantile_opt[0] - G_nc.qhat >= theta)

        for coeffs, d in G_nc.bounds.items():
            coeffs = np.array(coeffs)
            norm = coeffs[:-1]
            b = coeffs[-1]
            opt_model.addConstr(d * (norm[0] * merit_opt) <= d * b)

        opt_model.optimize()
        solve_time += opt_model.Runtime

        if opt_model.status == GRB.OPTIMAL:
            if opt_model.ObjVal < optimal_merit:
                optimal_merit = opt_model.ObjVal
                best_solution_found = True
                best_diagnostics = {
                    "route": "quantile_fallback",
                    "qhat": float(G_nc.qhat),
                }

    if best_solution_found:
        return (
            optimal_merit,
            True,
            solve_time,
            num_splits,
            "Optimal",
            best_diagnostics,
        )
    else:
        return (
            None,
            False,
            solve_time,
            num_splits,
            "NoCertifiedSolution",
            best_diagnostics,
        )


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
    feasibility_tol=1e-6,
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

        (
            optimal_merit,
            is_feasible,
            elapsed_time,
            num_splits,
            status,
            diagnostics,
        ) = run_enroll_rcc(
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
            feasibility_tol=feasibility_tol,
        )

        results.append(
            {
                "StudentID": student_id,
                "SAT": applicants_df.loc[i, "SAT"],
                "GPA": applicants_df.loc[i, "GPA"],
                "OptimalMerit": optimal_merit,
                "IsFeasible": is_feasible,
                "Status": status,
                "Route": diagnostics["route"],
                "LocalQhat": diagnostics["qhat"],
                "NumSplits": num_splits,
                "Time": elapsed_time,
                "Theta": theta,
            }
        )

        if (i + 1) % 100 == 0:
            logging.info(f"Processed {i + 1}/{len(applicants_df)} applicants")

    return pd.DataFrame(results)
