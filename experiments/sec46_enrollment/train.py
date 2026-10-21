"""Section 4.6.2: original two-stage training and artifact-saving entry point.

Only module imports and filesystem wiring have been reorganized.
"""

from conformal_cl.paths import PROJECT_ROOT
from conformal_cl.enrollment.training import train_main, train_quantile
import torch
import pickle
import pandas as pd

if __name__ == "__main__":
    # Run the basic regression training
    print("=" * 60)
    print("PART 1: Basic Regression Model")
    print("=" * 60)
    model, scaler, metrics = train_main()

    # Save the basic model, scaler, and metrics
    results_dir = PROJECT_ROOT / "results" / "enrollment"
    results_dir.mkdir(parents=True, exist_ok=True)

    model_path = results_dir / "enrollment_model_basic.pth"
    scaler_path = results_dir / "enrollment_scaler_basic.pkl"
    metrics_path = results_dir / "enrollment_metrics_basic.csv"

    torch.save(model.state_dict(), model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # Save metrics as CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_path, index=False)

    print(f"\nBasic model saved to '{model_path}'")
    print(f"Scaler saved to '{scaler_path}'")
    print(f"Metrics saved to '{metrics_path}'")

    # Run the quantile regression training
    # NOTE: This uses the basic model to generate probability predictions,
    # then trains quantiles on those probabilities (not on binary outcomes)
    print("\n" + "=" * 60)
    print("PART 2: Quantile Regression Model with Conformal Prediction")
    print("=" * 60)
    quantile_model, quantile_scaler, quantile_metrics, qhat = train_quantile(alpha=0.1)

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
