"""Small corrected enrollment RCC smoke-run entry point."""

from conformal_cl.paths import PROJECT_ROOT
from conformal_cl.enrollment.group_conformal import load_models_and_data, run_enroll_rcc_batch
import pandas as pd
import logging
import os
from datetime import datetime

if __name__ == "__main__":
    # Setup logging
    os.makedirs("logs", exist_ok=True)
    log_filename = os.path.join(
        "logs", f"enroll_RCC_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(),
        ],
    )

    logging.info(f"Starting Enrollment RCC execution. Log file: {log_filename}")

    # Load models and data
    logging.info("Loading models and calibration data...")
    quantile_model, basic_model, scaler, qhat, cal_X, cal_probs = load_models_and_data()
    logging.info(f"Loaded models. qhat = {qhat:.6f}")

    # Load applicant data
    data_path = PROJECT_ROOT / "data" / "college_applications6000.csv"
    df = pd.read_csv(data_path)
    logging.info(f"Loaded {len(df)} applicants")

    # Scale features
    features = df[["SAT", "GPA"]].values
    features_scaled = scaler.transform(features)

    # Run RCC optimization for a sample of applicants
    # Start with a small sample to test
    sample_size = 10
    df_sample = df.iloc[:sample_size].reset_index(drop=True)
    features_sample = features_scaled[:sample_size]

    logging.info(f"\nRunning RCC optimization for {sample_size} applicants...")
    logging.info("Parameters: theta=0.6, alpha=0.1, delta=0.1, max_split=64")

    results_df = run_enroll_rcc_batch(
        applicants_df=df_sample,
        features_scaled=features_sample,
        quantile_model=quantile_model,
        basic_model=basic_model,
        cal_X=cal_X,
        cal_probs=cal_probs,
        qhat_global=qhat,
        theta=0.6,
        alpha=0.1,
        delta=0.1,
        max_split=64,
        merit_lb=0.0,
        merit_ub=25.0,
    )

    # Save results
    results_path = (
        PROJECT_ROOT
        / "results"
        / "corrected"
        / "enrollment"
        / "enrollment_result_rcc_sample.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    logging.info(f"\nResults saved to {results_path}")

    # Print summary
    logging.info("\nSummary Statistics:")
    logging.info(f"  Total applicants: {len(results_df)}")
    logging.info(
        f"  Feasible solutions: {results_df['IsFeasible'].sum()} ({results_df['IsFeasible'].sum() / len(results_df) * 100:.1f}%)"
    )
    if results_df["IsFeasible"].sum() > 0:
        avg_merit = results_df[results_df["IsFeasible"]]["OptimalMerit"].mean() * 1000
        avg_splits = results_df[results_df["IsFeasible"]]["NumSplits"].mean()
        avg_time = results_df[results_df["IsFeasible"]]["Time"].mean()
        logging.info(f"  Average optimal merit: ${avg_merit:.2f}")
        logging.info(f"  Average number of splits: {avg_splits:.1f}")
        logging.info(f"  Average time per applicant: {avg_time:.4f} seconds")

    logging.info("\nEnrollment RCC execution completed")
