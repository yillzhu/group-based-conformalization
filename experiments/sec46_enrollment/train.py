"""Section 4.6.2: original two-stage training and artifact-saving entry point.

Only module imports and filesystem wiring have been reorganized.
"""

from conformal_cl.paths import PROJECT_ROOT
from conformal_cl.enrollment.training import train_main, train_quantile
from conformal_cl.provenance import write_run_manifest
import torch
import pickle
import pandas as pd

if __name__ == "__main__":
    # Corrected runs are intentionally separated from checked-in historical
    # artifacts under results/enrollment.
    results_dir = PROJECT_ROOT / "results" / "corrected" / "enrollment"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run the basic regression training
    print("=" * 60)
    print("PART 1: Basic Regression Model")
    print("=" * 60)
    model, scaler, metrics, split_ids = train_main(return_split_ids=True)

    # Save the basic model, scaler, and metrics
    model_path = results_dir / "enrollment_model_basic.pth"
    scaler_path = results_dir / "enrollment_scaler_basic.pkl"
    metrics_path = results_dir / "enrollment_metrics_basic.csv"

    torch.save(model.state_dict(), model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # Save metrics as CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_path, index=False)
    split_df = pd.concat(
        [
            pd.DataFrame({"StudentID": ids, "Split": split_name})
            for split_name, ids in split_ids.items()
        ],
        ignore_index=True,
    )
    split_ids_path = results_dir / "enrollment_split_ids.csv"
    split_df.to_csv(split_ids_path, index=False)

    print(f"\nBasic model saved to '{model_path}'")
    print(f"Scaler saved to '{scaler_path}'")
    print(f"Metrics saved to '{metrics_path}'")
    print(f"Split IDs saved to '{split_ids_path}'")

    # Run the quantile regression training
    # NOTE: This uses the basic model to generate probability predictions,
    # then trains quantiles on those probabilities (not on binary outcomes)
    print("\n" + "=" * 60)
    print("PART 2: Quantile Regression Model with Conformal Prediction")
    print("=" * 60)
    quantile_model, quantile_scaler, quantile_metrics, qhat = train_quantile(
        alpha=0.1,
        basic_model_path=model_path,
        basic_scaler_path=scaler_path,
    )

    # Save the quantile model, scaler, metrics, and qhat
    quantile_model_path = results_dir / "enrollment_model_quantile.pth"
    quantile_scaler_path = results_dir / "enrollment_scaler_quantile.pkl"
    quantile_metrics_path = results_dir / "enrollment_metrics_quantile.csv"
    qhat_path = results_dir / "enrollment_qhat.csv"

    torch.save(quantile_model.state_dict(), quantile_model_path)
    with open(quantile_scaler_path, "wb") as f:
        pickle.dump(quantile_scaler, f)

    # Save metrics as CSV
    quantile_metrics_df = pd.DataFrame([quantile_metrics])
    quantile_metrics_df.to_csv(quantile_metrics_path, index=False)

    # Save qhat as CSV
    qhat_df = pd.DataFrame({"qhat": [qhat], "alpha": [0.1]})
    qhat_df.to_csv(qhat_path, index=False)

    print(f"\nQuantile model saved to '{quantile_model_path}'")
    print(f"Scaler saved to '{quantile_scaler_path}'")
    print(f"Metrics saved to '{quantile_metrics_path}'")
    print(f"qhat saved to '{qhat_path}'")

    manifest_path = results_dir / "training_manifest.json"
    write_run_manifest(
        manifest_path,
        run_kind="corrected_enrollment_training",
        parameters={
            "random_state": 42,
            "basic_epochs": 100,
            "quantile_epochs": 1000,
            "alpha": 0.1,
            "split": {"train": 0.7, "test": 0.2, "calibration": 0.1},
            "scaler_fit": "training_rows_only",
        },
        inputs=[PROJECT_ROOT / "data" / "college_student_enroll-s1-1.csv"],
        artifacts=[
            model_path,
            scaler_path,
            metrics_path,
            split_ids_path,
            quantile_model_path,
            quantile_scaler_path,
            quantile_metrics_path,
            qhat_path,
        ],
    )
    print(f"Training provenance saved to '{manifest_path}'")
