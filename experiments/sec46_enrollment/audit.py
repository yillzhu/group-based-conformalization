"""Reproduce the enrollment example's non-solver validity diagnostics."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from conformal_cl.enrollment.audit import (
    binary_probability_metrics,
    monotonicity_summary,
    paired_decision_summary,
)
from conformal_cl.enrollment.models import create_enrollment_net, create_quantile_net
from conformal_cl.paths import PROJECT_ROOT


def _evaluate_grid(model, fixed_features, merit_grid, output_index=None, batch_size=250):
    rows = []
    with torch.no_grad():
        for start in range(0, len(fixed_features), batch_size):
            fixed = fixed_features[start : start + batch_size]
            inputs = np.empty((len(fixed), len(merit_grid), 3), dtype=np.float32)
            inputs[:, :, :2] = fixed[:, None, :]
            inputs[:, :, 2] = merit_grid
            outputs = model(torch.from_numpy(inputs.reshape(-1, 3))).numpy()
            if output_index is not None:
                outputs = outputs[:, output_index]
            rows.append(outputs.reshape(len(fixed), len(merit_grid)))
    return np.vstack(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "corrected" / "enrollment",
    )
    parser.add_argument("--grid-step", type=float, default=0.1)
    parser.add_argument("--monotonicity-tolerance", type=float, default=1e-6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.grid_step <= 0 or args.grid_step > 25:
        parser.error("--grid-step must be in (0, 25]")

    results = args.results_dir
    history = pd.read_csv(PROJECT_ROOT / "data" / "college_student_enroll-s1-1.csv")
    applicants = pd.read_csv(PROJECT_ROOT / "data" / "college_applications6000.csv")
    split_ids = pd.read_csv(results / "enrollment_split_ids.csv")
    test_ids = set(split_ids.loc[split_ids["Split"].eq("test"), "StudentID"])
    test = history[history["StudentID"].isin(test_ids)].copy()

    with open(results / "enrollment_scaler_basic.pkl", "rb") as stream:
        scaler = pickle.load(stream)
    basic = create_enrollment_net()
    basic.load_state_dict(
        torch.load(results / "enrollment_model_basic.pth", weights_only=True)
    )
    basic.eval()
    quantile = create_quantile_net()
    quantile.load_state_dict(
        torch.load(results / "enrollment_model_quantile.pth", weights_only=True)
    )
    quantile.eval()

    test_features = test[["SAT", "GPA", "merit"]].to_numpy(dtype=float)
    test_features[:, :2] = scaler.transform(test_features[:, :2])
    with torch.no_grad():
        test_probability = basic(
            torch.tensor(test_features, dtype=torch.float32)
        ).numpy()

    applicant_fixed = scaler.transform(
        applicants[["SAT", "GPA"]].to_numpy(dtype=float)
    )
    merit_grid = np.arange(
        0.0, 25.0 + args.grid_step / 2, args.grid_step, dtype=np.float32
    )
    basic_grid = _evaluate_grid(basic, applicant_fixed, merit_grid)
    lower_grid = _evaluate_grid(
        quantile, applicant_fixed, merit_grid, output_index=0
    )

    global_result = pd.read_csv(results / "enrollment_result_quantile.csv")
    rcc_result = pd.read_csv(results / "enrollment_result_rcc.csv")
    if not np.array_equal(global_result["StudentID"], rcc_result["StudentID"]):
        raise ValueError("global and RCC result rows are not aligned by StudentID")

    report = {
        "interpretation": {
            "data": "synthetic records published with the JANOS solver example",
            "point_target": "observed binary enrollment label",
            "interval_target": "prediction of the fitted point model",
            "rcc_calibration": "applicant-specific synthetic merit line",
            "oracle_response_available": False,
        },
        "held_out_point_model": binary_probability_metrics(
            test["enroll"].to_numpy(), test_probability
        ),
        "merit_monotonicity": {
            "grid_step_thousand_dollars": args.grid_step,
            "point_probability": monotonicity_summary(
                basic_grid, tolerance=args.monotonicity_tolerance
            ),
            "quantile_lower_endpoint": monotonicity_summary(
                lower_grid, tolerance=args.monotonicity_tolerance
            ),
        },
        "rcc_vs_global_merit_dollars": paired_decision_summary(
            1000 * rcc_result["OptimalMerit"].to_numpy(),
            1000 * global_result["OptimalMerit"].to_numpy(),
            equality_tolerance=1e-5,
        ),
    }

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
