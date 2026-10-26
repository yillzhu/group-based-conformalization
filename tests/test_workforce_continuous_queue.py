import unittest

import numpy as np

from conformal_cl.workforce.continuous_queue import (
    ContinuousQueueConfig,
    sample_continuous_queue_responses,
)
from conformal_cl.workforce.continuous_queue_experiment import (
    ContinuousQueueExperimentConfig,
    run_continuous_queue_experiment,
)


class ContinuousQueueTests(unittest.TestCase):
    def test_more_capacity_reduces_shortfall_under_common_scenarios(self):
        config = ContinuousQueueConfig()
        low = sample_continuous_queue_responses(
            np.tile([5.0, 3.0], (200, 1)),
            np.random.default_rng(12),
            config,
        )
        high = sample_continuous_queue_responses(
            np.tile([5.5, 3.5], (200, 1)),
            np.random.default_rng(12),
            config,
        )
        self.assertTrue(np.all(high <= low))

    def test_smoke_experiment_has_interior_oracle(self):
        result = run_continuous_queue_experiment(
            ContinuousQueueExperimentConfig(
                seed=2,
                training_size=200,
                calibration_size=300,
                evaluation_size=500,
                oracle_size=500,
                oracle_grid_size=101,
                quantile_iterations=100,
                max_splits=4,
            )
        )
        oracle = result["methods"]["oracle"]
        self.assertEqual(result["config"]["operational_alpha"], 0.1)
        self.assertEqual(result["config"]["calibration_alpha"], 0.1)
        self.assertEqual(oracle["status"], "selected")
        self.assertGreater(oracle["dedicated_fte_per_class"], 1.0)
        self.assertLess(oracle["dedicated_fte_per_class"], 12.0)
        self.assertGreater(oracle["flexible_fte"], 0.0)
        self.assertLess(oracle["flexible_fte"], 12.0)

    def test_spline_experiment_runs_over_continuous_domain(self):
        result = run_continuous_queue_experiment(
            ContinuousQueueExperimentConfig(
                seed=3,
                training_size=200,
                calibration_size=300,
                evaluation_size=500,
                oracle_size=500,
                oracle_grid_size=51,
                quantile_iterations=100,
                max_splits=4,
                diagnostic_size=200,
                dedicated_spline_knots=3,
            )
        )
        self.assertEqual(len(result["model"]["dedicated_spline_breakpoints"]), 3)
        self.assertEqual(result["methods"]["rcc"]["status"], "selected")
        self.assertIn("upper_pinball_loss", result["predictive_diagnostics"])


if __name__ == "__main__":
    unittest.main()
