"""Sections 4.6.1-4.6.2: applicant input, baseline and CQR merit optimization."""

from pathlib import Path
from conformal_cl.paths import PROJECT_ROOT
import pandas as pd
import numpy as np
import pickle
import gurobipy as gp
from gurobipy import Model, GRB
from gurobi_ml import add_predictor_constr
from .models import inverse_sigmoid
from conformal_cl.solver import gurobi_status_label, quiet_gurobi_env


def load_and_prepare_data(results_dir=None):
    """
    Load college applications data and prepare it for optimization.

    Returns:
        df: DataFrame with applicant data
        features_scaled: Scaled features (SAT, GPA) ready for model input
        scaler: The fitted scaler
    """
    # Paths
    data_path = PROJECT_ROOT / "data" / "college_applications6000.csv"
    if results_dir is None:
        results_dir = PROJECT_ROOT / "results" / "corrected" / "enrollment"
    scaler_path = Path(results_dir) / "enrollment_scaler_basic.pkl"

    # Load applicant data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} applicants")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst few rows:")
    print(df.head())

    # Load the scaler
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Extract and scale SAT and GPA
    features = df[["SAT", "GPA"]].values
    features_scaled = scaler.transform(features)

    return df, features_scaled, scaler


def optimize_merit_with_enrollment_constraint(
    applicant_idx, sat_scaled, gpa_scaled, model_nn, theta=0.6
):
    """
    Setup and solve a Gurobi optimization model to minimize merit aid
    subject to enrollment probability >= theta using the logit constraint.

    The learned neural network model (without sigmoid) outputs logit values.
    We optimize merit to achieve the minimum merit aid that ensures enrollment
    probability is at least theta by constraining logit(enrollment) >= logit(theta).

    Args:
        applicant_idx: Index of the applicant (for display purposes)
        sat_scaled: Scaled SAT score
        gpa_scaled: Scaled GPA
        model_nn: Trained PyTorch EnrollmentNet model WITHOUT sigmoid (outputs logits)
        theta: Minimum required enrollment probability (default 0.6)

    Returns:
        optimal_merit: Optimal merit aid amount
        enrollment_logit: Predicted enrollment logit at optimal merit
        model: The Gurobi model object
    """
    # Create Gurobi model
    m = gp.Model(
        f"enrollment_applicant_{applicant_idx}", env=quiet_gurobi_env()
    )

    # Decision variable: merit aid (bounded between 0 and 25000)
    merit = m.addVar(lb=0.0, ub=25.0, name="merit")

    # Create input variables for the neural network
    # The NN expects 3 inputs: [SAT_scaled, GPA_scaled, merit]
    # SAT and GPA are fixed (constants), merit is the decision variable
    sat_var = m.addVar(lb=sat_scaled, ub=sat_scaled, name="sat")
    gpa_var = m.addVar(lb=gpa_scaled, ub=gpa_scaled, name="gpa")

    # Create output variable for enrollment logit (pre-sigmoid value)
    enrollment_logit_var = m.addVar(
        lb=-GRB.INFINITY, ub=GRB.INFINITY, name="enroll_logit"
    )

    # Input features: [SAT, GPA, merit]
    input_vars = [sat_var, gpa_var, merit]
    output_vars = [enrollment_logit_var]

    # Embed the neural network as constraints using gurobi-ml
    add_predictor_constr(m, model_nn, input_vars, output_vars)

    # Compute the logit threshold
    theta_logit = inverse_sigmoid(theta)

    # Add constraint: logit(enrollment) >= logit(theta)
    # This is equivalent to P(enrollment) >= theta after sigmoid transformation
    m.addConstr(enrollment_logit_var >= theta_logit, name="min_enrollment_logit")

    # Objective: minimize merit aid
    m.setObjective(merit, GRB.MINIMIZE)

    # Solve the model
    m.optimize()

    # Extract solution
    if m.status == GRB.OPTIMAL:
        optimal_merit = merit.X
        enrollment_logit = enrollment_logit_var.X
        return optimal_merit, enrollment_logit, m
    else:
        print(f"Warning: Model status = {m.status}")
        return None, None, m


def optimize_merit_with_quantile_constraint(
    applicant_idx, sat_scaled, gpa_scaled, model_nn, qhat, theta=0.6
):
    """
    Setup and solve a Gurobi optimization model to minimize merit aid
    subject to lower quantile of enrollment probability (with conformal adjustment) >= theta.

    The learned quantile model predicts [lower, upper] quantiles for enrollment.
    We use the conformalized lower endpoint, ``lower quantile - qhat``, as the
    constraint.

    Args:
        applicant_idx: Index of the applicant (for display purposes)
        sat_scaled: Scaled SAT score
        gpa_scaled: Scaled GPA
        model_nn: Trained PyTorch QuantileNet model (outputs 2 values: [lower, upper])
        qhat: Conformal prediction adjustment value
        theta: Minimum required conformal lower endpoint (default 0.6)

    Returns:
        optimal_merit: Optimal merit aid amount
        lower_quantile_adjusted: Lower quantile - qhat at optimal merit
        model: The Gurobi model object
    """
    # Create Gurobi model
    m = gp.Model(
        f"enrollment_quantile_applicant_{applicant_idx}", env=quiet_gurobi_env()
    )

    # Decision variable: merit aid (bounded between 0 and 25)
    merit = m.addVar(lb=0.0, ub=25.0, name="merit")

    # Create input variables for the neural network
    sat_var = m.addVar(lb=sat_scaled, ub=sat_scaled, name="sat")
    gpa_var = m.addVar(lb=gpa_scaled, ub=gpa_scaled, name="gpa")

    # Create output variables for lower and upper quantiles
    lower_quantile_var = m.addVar(
        lb=-GRB.INFINITY, ub=GRB.INFINITY, name="lower_quantile"
    )
    upper_quantile_var = m.addVar(
        lb=-GRB.INFINITY, ub=GRB.INFINITY, name="upper_quantile"
    )

    # Input features: [SAT, GPA, merit]
    input_vars = [sat_var, gpa_var, merit]
    output_vars = [lower_quantile_var, upper_quantile_var]

    # Embed the neural network as constraints using gurobi-ml
    add_predictor_constr(m, model_nn, input_vars, output_vars)

    # The calibrated interval is [lower - qhat, upper + qhat].
    m.addConstr(lower_quantile_var - qhat >= theta, name="min_lower_quantile_conf")

    # Objective: minimize merit aid
    m.setObjective(merit, GRB.MINIMIZE)

    # Solve the model
    m.optimize()

    # Extract solution
    if m.status == GRB.OPTIMAL:
        optimal_merit = merit.X
        lower_quantile_adjusted = lower_quantile_var.X - qhat
        return optimal_merit, lower_quantile_adjusted, m
    else:
        return None, None, m
