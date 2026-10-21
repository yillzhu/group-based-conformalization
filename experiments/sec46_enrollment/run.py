"""Corrected enrollment comparison with optional bounded smoke-run settings."""

from pathlib import Path

from conformal_cl.paths import PROJECT_ROOT
from conformal_cl.enrollment.optimization import (
    load_and_prepare_data, optimize_merit_with_enrollment_constraint,
    optimize_merit_with_quantile_constraint, gurobi_status_label,
)
from conformal_cl.enrollment.models import (
    create_enrollment_net, create_enrollment_net_for_gurobi, create_quantile_net,
)
from conformal_cl.enrollment.group_conformal import run_enroll_rcc
from conformal_cl.enrollment.data import split_train_test_cal_indices
from conformal_cl.provenance import write_run_manifest
import argparse
import pandas as pd
import numpy as np
import pickle
import torch
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="run only the first N applicants")
    parser.add_argument("--theta", type=float, default=0.17)
    parser.add_argument("--max-split", type=int, default=64)
    parser.add_argument("--feasibility-tol", type=float, default=1e-6)
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "corrected" / "enrollment"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "corrected" / "enrollment"),
    )
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Enrollment Optimization with Gurobi")
    print("=" * 60)

    # Load and prepare data
    df, features_scaled, scaler = load_and_prepare_data(results_dir=model_dir)
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        df = df.iloc[: args.limit].reset_index(drop=True)
        features_scaled = features_scaled[: args.limit]

    # Load the trained neural network model WITH sigmoid for training
    model_path_trained = model_dir / "enrollment_model_basic.pth"

    # Create model WITH sigmoid to load trained weights
    model_with_sigmoid = create_enrollment_net(
        input_size=3, hidden_size=10, output_size=1
    )
    model_with_sigmoid.load_state_dict(torch.load(model_path_trained))
    model_with_sigmoid.eval()

    # Create Gurobi-compatible model WITHOUT sigmoid
    model_nn = create_enrollment_net_for_gurobi(
        input_size=3, hidden_size=10, output_size=1
    )

    # Copy the weights from the trained model (exclude sigmoid layer)
    # The trained model has layers: [Linear, ReLU, Linear, Sigmoid]
    # The Gurobi model has layers: [Linear, ReLU, Linear]
    model_nn[0].load_state_dict(model_with_sigmoid[0].state_dict())  # First Linear
    model_nn[1].load_state_dict(model_with_sigmoid[1].state_dict())  # ReLU
    model_nn[2].load_state_dict(model_with_sigmoid[2].state_dict())  # Second Linear
    model_nn.eval()

    print(
        "\nLoaded trained neural network model (Gurobi-compatible version without sigmoid)"
    )

    print("\n" + "=" * 60)
    print("MODEL 1: Minimize Merit with Enrollment Constraint")
    print("=" * 60)

    # Optimize for all applicants
    num_test = len(df)
    theta = args.theta  # Minimum enrollment probability

    print(f"\nOptimizing merit for all {num_test} applicants (theta={theta})...")

    # Store results
    results = []

    for i in tqdm(range(num_test), desc="Optimizing applicants", unit="applicant"):
        student_id = df.loc[i, "StudentID"]
        sat_orig = df.loc[i, "SAT"]
        gpa_orig = df.loc[i, "GPA"]
        sat_scaled = features_scaled[i, 0]
        gpa_scaled = features_scaled[i, 1]

        optimal_merit, enrollment_logit, model_obj = (
            optimize_merit_with_enrollment_constraint(
                student_id, sat_scaled, gpa_scaled, model_nn, theta
            )
        )

        if optimal_merit is not None:
            status = "Optimal"
            # Convert logit back to probability for reporting
            enrollment_prob = 1 / (1 + np.exp(-enrollment_logit))
            results.append(
                {
                    "StudentID": student_id,
                    "SAT": sat_orig,
                    "GPA": gpa_orig,
                    "OptimalMerit": optimal_merit,
                    "EnrollmentProb": enrollment_prob,
                    "Status": status,
                    "Theta": theta,
                }
            )
        else:
            status = gurobi_status_label(model_obj)
            results.append(
                {
                    "StudentID": student_id,
                    "SAT": sat_orig,
                    "GPA": gpa_orig,
                    "OptimalMerit": None,
                    "EnrollmentProb": None,
                    "Status": status,
                    "Theta": theta,
                }
            )

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_path = output_dir / "enrollment_result_basic.csv"
    results_df.to_csv(results_path, index=False)

    num_optimal = (results_df["Status"] == "Optimal").sum()
    num_infeasible = (results_df["Status"] == "Infeasible").sum()
    num_other = len(results_df) - num_optimal - num_infeasible
    avg_merit = results_df["OptimalMerit"].mean() * 1000  # Convert to dollars

    print(f"Optimization complete! Results saved to '{results_path}'")
    print("\nSummary:")
    print(f"  Total applicants optimized: {len(results_df)}")
    print(f"  Optimal solutions: {num_optimal}")
    print(f"  Infeasible solutions: {num_infeasible}")
    print(f"  Other solver outcomes: {num_other}")
    print(f"  Average optimal merit: ${avg_merit:.2f} (excluding infeasible)")

    print("\n" + "=" * 60)
    print("MODEL 1 Complete")
    print("=" * 60)

    # ============================================================
    # MODEL 2: Quantile Regression with Conformal Prediction
    # ============================================================

    print("\n" + "=" * 60)
    print("MODEL 2: Minimize Merit with Quantile Lower Bound Constraint")
    print("=" * 60)

    # Load the quantile model
    quantile_model_path = model_dir / "enrollment_model_quantile.pth"
    quantile_scaler_path = model_dir / "enrollment_scaler_quantile.pkl"
    qhat_path = model_dir / "enrollment_qhat.csv"

    # Load quantile model
    model_quantile = create_quantile_net(input_size=3, hidden_size=10, output_size=2)
    model_quantile.load_state_dict(torch.load(quantile_model_path))
    model_quantile.eval()

    # Load quantile scaler
    with open(quantile_scaler_path, "rb") as f:
        scaler_quantile = pickle.load(f)

    # Load qhat
    qhat_df = pd.read_csv(qhat_path)
    qhat = qhat_df.loc[0, "qhat"]

    print(f"\nLoaded quantile model and qhat = {qhat:.6f}")

    # Scale features with quantile scaler
    features_quantile = df[["SAT", "GPA"]].values
    features_scaled_quantile = scaler_quantile.transform(features_quantile)

    print(f"\nOptimizing merit for all {num_test} applicants (theta={theta})...")

    # Store results
    results_quantile = []

    for i in tqdm(
        range(num_test), desc="Optimizing applicants (quantile)", unit="applicant"
    ):
        student_id = df.loc[i, "StudentID"]
        sat_orig = df.loc[i, "SAT"]
        gpa_orig = df.loc[i, "GPA"]
        sat_scaled = features_scaled_quantile[i, 0]
        gpa_scaled = features_scaled_quantile[i, 1]

        optimal_merit, lower_quantile_adj, model_obj = (
            optimize_merit_with_quantile_constraint(
                student_id, sat_scaled, gpa_scaled, model_quantile, qhat, theta
            )
        )

        if optimal_merit is not None:
            status = "Optimal"
            results_quantile.append(
                {
                    "StudentID": student_id,
                    "SAT": sat_orig,
                    "GPA": gpa_orig,
                    "OptimalMerit": optimal_merit,
                    "LowerQuantileAdjusted": lower_quantile_adj,
                    "Status": status,
                    "Theta": theta,
                }
            )
        else:
            status = gurobi_status_label(model_obj)
            results_quantile.append(
                {
                    "StudentID": student_id,
                    "SAT": sat_orig,
                    "GPA": gpa_orig,
                    "OptimalMerit": None,
                    "LowerQuantileAdjusted": None,
                    "Status": status,
                    "Theta": theta,
                }
            )

    # Save results to CSV
    results_quantile_df = pd.DataFrame(results_quantile)
    results_quantile_path = output_dir / "enrollment_result_quantile.csv"
    results_quantile_df.to_csv(results_quantile_path, index=False)

    num_optimal_q = (results_quantile_df["Status"] == "Optimal").sum()
    num_infeasible_q = (results_quantile_df["Status"] == "Infeasible").sum()
    num_other_q = len(results_quantile_df) - num_optimal_q - num_infeasible_q
    avg_merit_q = results_quantile_df["OptimalMerit"].mean() * 1000

    print(f"Optimization complete! Results saved to '{results_quantile_path}'")
    print("\nSummary:")
    print(f"  Total applicants optimized: {len(results_quantile_df)}")
    print(f"  Optimal solutions: {num_optimal_q}")
    print(f"  Infeasible solutions: {num_infeasible_q}")
    print(f"  Other solver outcomes: {num_other_q}")
    print(f"  Average optimal merit: ${avg_merit_q:.2f} (excluding infeasible)")

    # ============================================================
    # MODEL 3: RCC Approach
    # ============================================================

    print("\n" + "=" * 60)
    print("MODEL 3: RCC Approach with Adaptive Group Splitting")
    print("=" * 60)

    # Prepare calibration data for RCC
    # Load calibration data and probabilities
    data_path_train = (
        PROJECT_ROOT / "data" / "college_student_enroll-s1-1.csv"
    )
    df_train = pd.read_csv(data_path_train)
    features_train = df_train[["SAT", "GPA", "merit"]].values
    features_train[:, :2] = scaler_quantile.transform(features_train[:, :2])

    split_ids_path = model_dir / "enrollment_split_ids.csv"
    if split_ids_path.exists():
        split_ids_df = pd.read_csv(split_ids_path)
        calibration_ids = set(
            split_ids_df.loc[
                split_ids_df["Split"].eq("calibration"), "StudentID"
            ]
        )
        calibration_mask = df_train["StudentID"].isin(calibration_ids)
        cal_X = features_train[calibration_mask.to_numpy()]
    else:
        _, _, calibration_idx = split_train_test_cal_indices(len(df_train))
        cal_X = features_train[calibration_idx]

    # Load basic model to get calibration probabilities
    model_basic_for_cal = create_enrollment_net(
        input_size=3, hidden_size=10, output_size=1
    )
    model_basic_for_cal.load_state_dict(
        torch.load(
            model_dir / "enrollment_model_basic.pth"
        )
    )
    model_basic_for_cal.eval()

    with torch.no_grad():
        cal_probs = model_basic_for_cal(torch.FloatTensor(cal_X)).numpy()

    print(f"\nCalibration data: {len(cal_X)} samples")
    print(f"Running RCC optimization for all {num_test} applicants (theta={theta})...")
    print(f"Parameters: alpha=0.1, delta=0.1, max_split={args.max_split}")

    # Store results
    results_rcc = []

    for i in tqdm(
        range(num_test), desc="Optimizing applicants (RCC)", unit="applicant"
    ):
        student_id = df.loc[i, "StudentID"]
        sat_orig = df.loc[i, "SAT"]
        gpa_orig = df.loc[i, "GPA"]
        sat_scaled = features_scaled_quantile[i, 0]
        gpa_scaled = features_scaled_quantile[i, 1]

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
            quantile_model=model_quantile,
            basic_model=model_basic_for_cal,
            cal_X=cal_X,
            cal_probs=cal_probs,
            qhat_global=qhat,
            theta=theta,
            alpha=0.1,
            delta=0.1,
            max_split=args.max_split,
            merit_lb=0.0,
            merit_ub=25.0,
            feasibility_tol=args.feasibility_tol,
        )

        if is_feasible:
            status = "Optimal"
            results_rcc.append(
                {
                    "StudentID": student_id,
                    "SAT": sat_orig,
                    "GPA": gpa_orig,
                    "OptimalMerit": optimal_merit,
                    "NumSplits": num_splits,
                    "Route": diagnostics["route"],
                    "LocalQhat": diagnostics["qhat"],
                    "Status": status,
                    "Theta": theta,
                }
            )
        else:
            results_rcc.append(
                {
                    "StudentID": student_id,
                    "SAT": sat_orig,
                    "GPA": gpa_orig,
                    "OptimalMerit": None,
                    "NumSplits": num_splits,
                    "Route": diagnostics["route"],
                    "LocalQhat": diagnostics["qhat"],
                    "Status": status,
                    "Theta": theta,
                }
            )

    # Save results to CSV
    results_rcc_df = pd.DataFrame(results_rcc)
    results_rcc_path = output_dir / "enrollment_result_rcc.csv"
    results_rcc_df.to_csv(results_rcc_path, index=False)

    num_optimal_rcc = (results_rcc_df["Status"] == "Optimal").sum()
    num_infeasible_rcc = (results_rcc_df["Status"] == "Infeasible").sum()
    num_other_rcc = len(results_rcc_df) - num_optimal_rcc - num_infeasible_rcc
    avg_merit_rcc = (
        results_rcc_df[results_rcc_df["Status"] == "Optimal"]["OptimalMerit"].mean()
        * 1000
        if num_optimal_rcc > 0
        else 0
    )
    avg_splits = (
        results_rcc_df[results_rcc_df["Status"] == "Optimal"]["NumSplits"].mean()
        if num_optimal_rcc > 0
        else 0
    )

    print(f"Optimization complete! Results saved to '{results_rcc_path}'")
    print("\nSummary:")
    print(f"  Total applicants optimized: {len(results_rcc_df)}")
    print(f"  Optimal solutions: {num_optimal_rcc}")
    print(f"  Infeasible solutions: {num_infeasible_rcc}")
    print(f"  Other solver outcomes: {num_other_rcc}")
    print(f"  Average optimal merit: ${avg_merit_rcc:.2f} (excluding infeasible)")
    print(f"  Average number of splits: {avg_splits:.1f}")

    print("\n" + "=" * 60)
    print("MODEL 3 Complete")
    print("=" * 60)

    # ============================================================
    # Comparison Summary
    # ============================================================

    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)

    # Create comparison dataframe
    comparison_data = {
        "Method": ["Basic Model", "Quantile Model", "RCC Model"],
        "Total_Applicants": [
            len(results_df),
            len(results_quantile_df),
            len(results_rcc_df),
        ],
        "Optimal_Solutions": [num_optimal, num_optimal_q, num_optimal_rcc],
        "Infeasible_Solutions": [num_infeasible, num_infeasible_q, num_infeasible_rcc],
        "Other_Solver_Outcomes": [num_other, num_other_q, num_other_rcc],
        "Average_Merit_Dollars": [avg_merit, avg_merit_q, avg_merit_rcc],
        "Theta": [theta, theta, theta],
    }

    comparison_df = pd.DataFrame(comparison_data)

    # Save comparison to CSV
    comparison_path = output_dir / "enrollment_result_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    print(f"\nComparison saved to '{comparison_path}'")
    print("\nMethod Comparison:")
    print(f"{'Method':<35} {'Avg Merit':<15} {'Difference':<15}")
    print("-" * 65)
    print(f"{'Basic Model':<35} ${avg_merit:<14.2f} -")
    print(
        f"{'Quantile Model (Conservative)':<35} ${avg_merit_q:<14.2f} +${avg_merit_q - avg_merit:.2f}"
    )
    print(
        f"{'RCC Model (Adaptive)':<35} ${avg_merit_rcc:<14.2f} +${avg_merit_rcc - avg_merit:.2f}"
    )

    # Calculate additional statistics
    baseline_merit = results_df.loc[
        results_df["Status"].eq("Optimal"), ["StudentID", "OptimalMerit"]
    ]

    quantile_pairs = baseline_merit.merge(
        results_quantile_df.loc[
            results_quantile_df["Status"].eq("Optimal"),
            ["StudentID", "OptimalMerit"],
        ],
        on="StudentID",
        how="inner",
        validate="one_to_one",
        suffixes=("_basic", "_quantile"),
    )
    merit_increases_q = (
        quantile_pairs["OptimalMerit_quantile"]
        > quantile_pairs["OptimalMerit_basic"]
    ).sum()
    merit_same_q = (
        quantile_pairs["OptimalMerit_quantile"]
        == quantile_pairs["OptimalMerit_basic"]
    ).sum()

    rcc_pairs = baseline_merit.merge(
        results_rcc_df.loc[
            results_rcc_df["Status"].eq("Optimal"),
            ["StudentID", "OptimalMerit"],
        ],
        on="StudentID",
        how="inner",
        validate="one_to_one",
        suffixes=("_basic", "_rcc"),
    )
    merit_increases_rcc = (
        rcc_pairs["OptimalMerit_rcc"] > rcc_pairs["OptimalMerit_basic"]
    ).sum()
    merit_same_rcc = (
        rcc_pairs["OptimalMerit_rcc"] == rcc_pairs["OptimalMerit_basic"]
    ).sum()

    print("\nDetailed Comparison:")
    print(f"\n  Quantile Model vs Basic Model:")
    if len(quantile_pairs):
        print(
            f"    Cases where quantile requires MORE merit: {merit_increases_q} "
            f"({merit_increases_q / len(quantile_pairs) * 100:.1f}% of paired optimal solutions)"
        )
        print(
            f"    Cases where both require SAME merit: {merit_same_q} "
            f"({merit_same_q / len(quantile_pairs) * 100:.1f}% of paired optimal solutions)"
        )
    else:
        print("    No paired optimal quantile/basic solutions were available.")
    print(f"    Additional cost: ${avg_merit_q - avg_merit:.2f} per applicant")

    print(f"\n  RCC Model vs Basic Model:")
    if len(rcc_pairs):
        print(
            f"    Cases where RCC requires MORE merit: {merit_increases_rcc} "
            f"({merit_increases_rcc / len(rcc_pairs) * 100:.1f}% of paired optimal solutions)"
        )
        print(
            f"    Cases where both require SAME merit: {merit_same_rcc} "
            f"({merit_same_rcc / len(rcc_pairs) * 100:.1f}% of paired optimal solutions)"
        )
    else:
        print("    No paired optimal RCC/basic solutions were available.")
    print(f"    Additional cost: ${avg_merit_rcc - avg_merit:.2f} per applicant")

    print("\n" + "=" * 60)
    print("All Optimizations Complete")
    print("=" * 60)

    manifest_path = output_dir / "optimization_manifest.json"
    write_run_manifest(
        manifest_path,
        run_kind="corrected_enrollment_optimization",
        parameters={
            "theta": theta,
            "alpha": 0.1,
            "delta": 0.1,
            "max_split": args.max_split,
            "applicant_limit": args.limit,
            "merit_bounds_thousands": [0.0, 25.0],
            "feasibility_tolerance": args.feasibility_tol,
            "timing": "sum_of_gurobi_runtime_seconds",
        },
        inputs=[
            PROJECT_ROOT / "data" / "college_applications6000.csv",
            PROJECT_ROOT / "data" / "college_student_enroll-s1-1.csv",
            model_path_trained,
            quantile_model_path,
            quantile_scaler_path,
            qhat_path,
            split_ids_path,
        ],
        artifacts=[
            results_path,
            results_quantile_path,
            results_rcc_path,
            comparison_path,
        ],
    )
    print(f"Optimization provenance saved to '{manifest_path}'")
