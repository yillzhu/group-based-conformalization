import unittest

import numpy as np

from conformal_cl.enrollment.audit import (
    binary_probability_metrics,
    monotonicity_summary,
    paired_decision_summary,
)


class EnrollmentAuditTests(unittest.TestCase):
    def test_probability_metrics_perfect_predictions(self):
        metrics = binary_probability_metrics([0, 0, 1, 1], [0, 0, 1, 1])
        self.assertEqual(metrics["brier_score"], 0.0)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["ece_10_equal_width"], 0.0)

    def test_monotonicity_summary_counts_rows(self):
        summary = monotonicity_summary([[0, 1, 2], [0, 2, 1]])
        self.assertEqual(summary["violating_rows"], 1)
        self.assertAlmostEqual(summary["violation_rate"], 0.5)
        self.assertEqual(summary["minimum_step"], -1.0)

    def test_paired_decision_summary(self):
        summary = paired_decision_summary([1, 2, 5], [2, 2, 3])
        self.assertEqual(summary["candidate_cheaper"], 1)
        self.assertEqual(summary["equal"], 1)
        self.assertEqual(summary["candidate_costlier"], 1)
        self.assertAlmostEqual(summary["mean_absolute_difference"], 1.0)

    def test_rejects_invalid_probabilities(self):
        with self.assertRaises(ValueError):
            binary_probability_metrics([0, 1], [0.2, np.nan])


if __name__ == "__main__":
    unittest.main()
