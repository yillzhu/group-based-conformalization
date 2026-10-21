"""Section 4.6.2: randomized historical train/test/calibration split.

Historical implementation; known issues are deliberately retained.
See docs/findings.md and KNOWN-ISSUE markers.
"""

from sklearn.model_selection import train_test_split


def split_train_test_cal(
    features, labels, test_size=0.2, cal_size=0.1, random_state=42
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
    # First split: separate out test set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )

    # Second split: from remaining 80%, split into train (87.5% of 80% = 70% total)
    # and cal (12.5% of 80% = 10% total)
    cal_size_adjusted = cal_size / (1 - test_size)  # 0.1 / 0.8 = 0.125
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_temp, y_temp, test_size=cal_size_adjusted, random_state=random_state
    )

    return X_train, X_test, X_cal, y_train, y_test, y_cal
