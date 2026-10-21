import unittest

import numpy as np

from conformal_cl.order_stats import (
    InsufficientCalibrationError,
    conformal_order_statistic,
)


class ConformalOrderStatisticTests(unittest.TestCase):
    def test_selects_exact_one_based_rank(self):
        scores = np.arange(1, 11, dtype=float)

        # ceil((10 + 1) * .8) = 9, so the ninth value is required.
        self.assertEqual(
            conformal_order_statistic(scores, alpha=0.2),
            9.0,
        )

    def test_is_invariant_to_shape_and_input_order(self):
        scores = np.array([[4.0], [1.0], [3.0], [2.0], [5.0]])
        self.assertEqual(
            conformal_order_statistic(scores, alpha=0.4),
            4.0,
        )

    def test_rejects_unavailable_rank(self):
        with self.assertRaises(InsufficientCalibrationError):
            conformal_order_statistic([1.0, 2.0], alpha=0.1)

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            conformal_order_statistic([1.0], alpha=0)
        with self.assertRaises(ValueError):
            conformal_order_statistic([np.nan], alpha=0.1)


if __name__ == "__main__":
    unittest.main()
