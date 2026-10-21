import unittest

import numpy as np

from conformal_cl.enrollment.data import (
    split_train_test_cal,
    split_train_test_cal_indices,
)


class EnrollmentSplitTests(unittest.TestCase):
    def test_indices_are_deterministic_disjoint_and_complete(self):
        first = split_train_test_cal_indices(100, random_state=42)
        second = split_train_test_cal_indices(100, random_state=42)
        self.assertEqual(first, second)

        train, test, calibration = map(set, first)
        self.assertFalse(train & test)
        self.assertFalse(train & calibration)
        self.assertFalse(test & calibration)
        self.assertEqual(train | test | calibration, set(range(100)))
        self.assertEqual((len(train), len(test), len(calibration)), (70, 20, 10))

    def test_split_returns_corresponding_features_and_labels(self):
        features = np.arange(300).reshape(100, 3)
        labels = np.arange(100).reshape(100, 1)
        *parts, indices = split_train_test_cal(
            features, labels, random_state=7, return_indices=True
        )
        x_train, x_test, x_cal, y_train, y_test, y_cal = parts
        train_idx, test_idx, cal_idx = indices
        np.testing.assert_array_equal(x_train, features[train_idx])
        np.testing.assert_array_equal(x_test, features[test_idx])
        np.testing.assert_array_equal(x_cal, features[cal_idx])
        np.testing.assert_array_equal(y_train, labels[train_idx])
        np.testing.assert_array_equal(y_test, labels[test_idx])
        np.testing.assert_array_equal(y_cal, labels[cal_idx])


if __name__ == "__main__":
    unittest.main()
