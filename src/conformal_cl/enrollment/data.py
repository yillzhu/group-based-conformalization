"""Section 4.6.2: deterministic train/test/calibration split helpers."""

from sklearn.model_selection import train_test_split


def split_train_test_cal_indices(
    n_rows, test_size=0.2, cal_size=0.1, random_state=42
):
    """Return deterministic row indices for the train/test/calibration split."""

    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    indices = list(range(n_rows))
    temp_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )
    cal_size_adjusted = cal_size / (1 - test_size)
    train_idx, cal_idx = train_test_split(
        temp_idx, test_size=cal_size_adjusted, random_state=random_state
    )
    return train_idx, test_idx, cal_idx


def split_train_test_cal(
    features,
    labels,
    test_size=0.2,
    cal_size=0.1,
    random_state=42,
    return_indices=False,
):
    """
    Split data into 70% train, 20% test, 10% calibration sets.

    Args:
        features: Feature array
        labels: Label array
        test_size: Proportion for test set (default 0.2)
        cal_size: Proportion for calibration set (default 0.1)
        random_state: Random seed

    Returns:
        X_train, X_test, X_cal, y_train, y_test, y_cal
    """
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same number of rows")

    train_idx, test_idx, cal_idx = split_train_test_cal_indices(
        len(features),
        test_size=test_size,
        cal_size=cal_size,
        random_state=random_state,
    )
    result = (
        features[train_idx],
        features[test_idx],
        features[cal_idx],
        labels[train_idx],
        labels[test_idx],
        labels[cal_idx],
    )
    if return_indices:
        return (*result, (train_idx, test_idx, cal_idx))
    return result
