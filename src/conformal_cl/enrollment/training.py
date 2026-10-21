"""Section 4.6.2: classifier fitting and quantile fitting to classifier predictions.

Historical implementation; known issues are deliberately retained.
See docs/findings.md and KNOWN-ISSUE markers.
"""

from pathlib import Path
from conformal_cl.paths import PROJECT_ROOT
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from .data import split_train_test_cal
from .calibration import conformalizing
from .models import create_enrollment_net, create_quantile_net, interval_score_loss


def train_main(
    csv_path=None,
    epochs=100,
    learning_rate=0.01,
    test_size=0.2,
    random_state=42,
):
    """
    Train a one-layer ReLU network with 10 hidden neurons on enrollment data.

    Args:
        csv_path: Path to the CSV file
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility

    Returns:
        model: Trained neural network model
        scaler: Fitted StandardScaler
        metrics: Dictionary of evaluation metrics
    """

    # Default path relative to script location
    if csv_path is None:
        csv_path = (
            PROJECT_ROOT / "data" / "college_student_enroll-s1-1.csv"
        )

    # Read the CSV file
    df = pd.read_csv(csv_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst few rows:")
    print(df.head())

    # Extract features and labels
    features = df[["SAT", "GPA", "merit"]].values
    labels = df["enroll"].values.reshape(-1, 1)

    # Scale SAT and GPA columns using StandardScaler
    # KNOWN-ISSUE[ISSUE-04]: scaler is fitted before splitting, on all rows.
    scaler = StandardScaler()
    features[:, :2] = scaler.fit_transform(
        features[:, :2]
    )  # Scale SAT and GPA (first two columns)

    print(
        f"\nScaled features - Mean: {features[:, :2].mean(axis=0)}, Std: {features[:, :2].std(axis=0)}"
    )

    # Split data into train (70%), test (20%), and calibration (10%) sets
    X_train, X_test, X_cal, y_train, y_test, y_cal = split_train_test_cal(
        features, labels, test_size=0.2, cal_size=0.1, random_state=random_state
    )

    print(
        f"Train size: {len(X_train)} (70%), Test size: {len(X_test)} (20%), Cal size: {len(X_cal)} (10%)"
    )

    # Convert to PyTorch tensors (only train and test for this model)
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    # Initialize the model (with sigmoid for classification)
    # KNOWN-ISSUE[ISSUE-09]: no torch seed is added in this refactor.
    model = create_enrollment_net(input_size=3, hidden_size=10, output_size=1)

    # Loss function and optimizer (Binary Cross Entropy for classification)
    criterion = nn.BCELoss()  # Expects outputs in [0, 1] and labels in {0, 1}
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()

        # Forward pass (outputs are probabilities due to sigmoid)
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                test_loss = criterion(test_outputs, y_test_tensor)
            print(
                f"Epoch [{epoch + 1}/{epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}"
            )

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        train_predictions = model(X_train_tensor).numpy()
        test_predictions = model(X_test_tensor).numpy()

    # Calculate metrics (predictions are probabilities in [0, 1])
    train_mse = mean_squared_error(y_train, train_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)
    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)

    # Calculate accuracy for binary classification (threshold at 0.5)
    train_accuracy = ((train_predictions >= 0.5) == y_train).mean()
    test_accuracy = ((test_predictions >= 0.5) == y_test).mean()

    metrics = {
        "train_mse": train_mse,
        "test_mse": test_mse,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
    }

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(
        f"Train MSE: {train_mse:.4f}, Train R²: {train_r2:.4f}, Train Accuracy: {train_accuracy:.4f}"
    )
    print(
        f"Test MSE: {test_mse:.4f}, Test R²: {test_r2:.4f}, Test Accuracy: {test_accuracy:.4f}"
    )
    print(
        f"Predicted probabilities - Min: {train_predictions.min():.4f}, Max: {train_predictions.max():.4f}, Mean: {train_predictions.mean():.4f}"
    )

    return model, scaler, metrics


def train_quantile(
    csv_path=None,
    basic_model_path=None,
    alpha=0.1,
    epochs=1000,
    learning_rate=0.01,
    test_size=0.2,
    random_state=42,
):
    """
    Train a two-layer ReLU network (10 neurons each) for quantile regression on probability predictions.

    This uses the post-sigmoid probability outputs from the basic classification model as labels.

    This function trains a quantile model on PREDICTED PROBABILITIES from the basic model,
    rather than on binary outcomes. This gives us quantiles of the probability distribution,
    which is more appropriate for conservative decision-making.

    Args:
        csv_path: Path to the CSV file
        basic_model_path: Path to the trained basic model (if None, will look in default location)
        alpha: Miscoverage level (default 0.1 for (1-alpha)=90% prediction interval)
        epochs: Number of training epochs (default 1000, same as reference)
        learning_rate: Learning rate for optimizer
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility

    Returns:
        model: Trained quantile regression model
        scaler: Fitted StandardScaler
        metrics: Dictionary with interval coverage and width
        qhat: Conformal prediction adjustment
    """

    # Default paths
    if csv_path is None:
        csv_path = (
            PROJECT_ROOT / "data" / "college_student_enroll-s1-1.csv"
        )

    if basic_model_path is None:
        basic_model_path = (
            PROJECT_ROOT
            / "results"
            / "enrollment"
            / "enrollment_model_basic.pth"
        )
        # KNOWN-ISSUE[ISSUE-09]: undefined outside the default-model-path branch.
        basic_scaler_path = (
            PROJECT_ROOT
            / "results"
            / "enrollment"
            / "enrollment_scaler_basic.pkl"
        )

    print(f"\n{'=' * 50}")
    print(f"Training Quantile Regression on Predicted Probabilities (alpha={alpha})")
    print(f"{'=' * 50}")

    # Load the basic model to generate probability predictions
    print("Loading basic model to generate probability predictions...")
    basic_model = create_enrollment_net(input_size=3, hidden_size=10, output_size=1)
    basic_model.load_state_dict(torch.load(basic_model_path))
    basic_model.eval()

    with open(basic_scaler_path, "rb") as f:
        basic_scaler = pickle.load(f)

    # Read the CSV file
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")

    # Extract features and scale them using the BASIC model's scaler
    features = df[["SAT", "GPA", "merit"]].values
    features_scaled = features.copy()
    features_scaled[:, :2] = basic_scaler.transform(features[:, :2])

    # KNOWN-ISSUE[ISSUE-05]: targets are teacher predictions, not true probabilities.
    # Generate predicted probabilities from basic model as new labels
    print("Generating predicted probabilities from basic model...")
    X_tensor = torch.FloatTensor(features_scaled)
    with torch.no_grad():
        predicted_probs = basic_model(X_tensor).numpy()

    print(
        f"Predicted probabilities - Min: {predicted_probs.min():.4f}, Max: {predicted_probs.max():.4f}, Mean: {predicted_probs.mean():.4f}"
    )

    # Use the SAME scaler as basic model for consistency
    scaler = basic_scaler

    # Split data into train (70%), test (20%), and calibration (10%) sets
    # Now using predicted probabilities as labels instead of binary outcomes
    X_train, X_test, X_cal, y_train, y_test, y_cal = split_train_test_cal(
        features_scaled,
        predicted_probs,
        test_size=0.2,
        cal_size=0.1,
        random_state=random_state,
    )

    print(
        f"Train size: {len(X_train)} (70%), Test size: {len(X_test)} (20%), Cal size: {len(X_cal)} (10%)"
    )

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    X_cal_tensor = torch.FloatTensor(X_cal)

    # Initialize the quantile model (outputs 2 values: lower and upper quantiles)
    model = create_quantile_net(input_size=3, hidden_size=10, output_size=2)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()

        # Forward pass
        outputs = model(X_train_tensor)
        loss = interval_score_loss(outputs, y_train_tensor, alpha)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                test_loss = interval_score_loss(test_outputs, y_test_tensor, alpha)
            print(
                f"Epoch [{epoch + 1}/{epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}"
            )

    # Evaluate the model (before conformal adjustment)
    model.eval()
    with torch.no_grad():
        train_predictions = model(X_train_tensor).numpy()
        test_predictions = model(X_test_tensor).numpy()

    # Calculate coverage and interval width (before conformal adjustment)
    train_lower = train_predictions[:, 0:1]
    train_upper = train_predictions[:, 1:2]
    test_lower = test_predictions[:, 0:1]
    test_upper = test_predictions[:, 1:2]

    train_coverage_before = ((y_train >= train_lower) & (y_train <= train_upper)).mean()
    test_coverage_before = ((y_test >= test_lower) & (y_test <= test_upper)).mean()

    train_width_before = (train_upper - train_lower).mean()
    test_width_before = (test_upper - test_lower).mean()

    print("\n" + "=" * 50)
    print("Before Conformal Adjustment:")
    print("=" * 50)
    print(f"Target Coverage: {1 - alpha:.1%}")
    print(
        f"Train Coverage: {train_coverage_before:.4f} ({train_coverage_before:.1%}), Width: {train_width_before:.4f}"
    )
    print(
        f"Test Coverage: {test_coverage_before:.4f} ({test_coverage_before:.1%}), Width: {test_width_before:.4f}"
    )

    # Conformal prediction calibration
    print("\n" + "=" * 50)
    print("Applying Conformal Calibration...")
    print("=" * 50)
    qhat = conformalizing(model, X_cal_tensor, y_cal, alpha)
    print(f"Conformal adjustment (qhat): {qhat:.4f}")

    # Apply conformal adjustment to predictions
    train_lower_conf = train_lower - qhat
    train_upper_conf = train_upper + qhat
    test_lower_conf = test_lower - qhat
    test_upper_conf = test_upper + qhat

    # Calculate coverage and interval width (after conformal adjustment)
    train_coverage_after = (
        (y_train >= train_lower_conf) & (y_train <= train_upper_conf)
    ).mean()
    test_coverage_after = (
        (y_test >= test_lower_conf) & (y_test <= test_upper_conf)
    ).mean()

    train_width_after = (train_upper_conf - train_lower_conf).mean()
    test_width_after = (test_upper_conf - test_lower_conf).mean()

    metrics = {
        "train_coverage_before": train_coverage_before,
        "test_coverage_before": test_coverage_before,
        "train_width_before": train_width_before,
        "test_width_before": test_width_before,
        "train_coverage_after": train_coverage_after,
        "test_coverage_after": test_coverage_after,
        "train_width_after": train_width_after,
        "test_width_after": test_width_after,
        "qhat": qhat,
        "target_coverage": 1 - alpha,
    }

    print("\n" + "=" * 50)
    print("After Conformal Adjustment:")
    print("=" * 50)
    print(
        f"Train Coverage: {train_coverage_after:.4f} ({train_coverage_after:.1%}), Width: {train_width_after:.4f}"
    )
    print(
        f"Test Coverage: {test_coverage_after:.4f} ({test_coverage_after:.1%}), Width: {test_width_after:.4f}"
    )

    return model, scaler, metrics, qhat
