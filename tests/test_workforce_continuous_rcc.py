import unittest

import numpy as np

from conformal_cl.workforce.continuous_rcc import (
    clip_polygon,
    run_continuous_rcc,
)
from conformal_cl.workforce.models import LinearModel
from conformal_cl.workforce.continuous_experiment import (
    ContinuousExperimentConfig,
    run_continuous_experiment,
    summarize_continuous_runs,
)


class ContinuousRCCTests(unittest.TestCase):
    def test_polygon_clipping_keeps_continuous_intersection(self):
        square = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        clipped = clip_polygon(square, np.asarray([1.0, 1.0]), 1.3)
        self.assertTrue(np.any(np.all(np.isclose(clipped, [1.0, 0.3]), axis=1)))
        self.assertTrue(np.all(clipped.sum(axis=1) <= 1.3 + 1e-9))

    def test_literal_rcc_optimizes_over_continuous_regions(self):
        center = np.zeros(2)
        scale = np.ones(2)
        # Both endpoints equal 1.3 - d - f, making all CQR scores zero.
        endpoint = LinearModel(np.asarray([1.3, -1.0, -1.0]), center, scale)
        rng = np.random.default_rng(9)
        calibration_x = rng.uniform(0.0, 1.0, size=(1000, 2))
        calibration_y = endpoint.predict(calibration_x)
        result = run_continuous_rcc(
            endpoint,
            endpoint,
            calibration_x,
            calibration_y,
            domain_lower=np.asarray([0.0, 0.0]),
            domain_upper=np.asarray([1.0, 1.0]),
            max_splits=8,
        )
        self.assertEqual(result.status, "selected")
        self.assertGreater(result.splits, 0)
        np.testing.assert_allclose(result.decision, [1.0, 0.3], atol=1e-8)
        self.assertAlmostEqual(result.objective, 2.72)

    def test_continuous_experiment_smoke_run(self):
        result = run_continuous_experiment(
            ContinuousExperimentConfig(
                seed=4,
                training_size=200,
                calibration_size=300,
                evaluation_size=500,
                oracle_size=500,
                oracle_grid_size=51,
                quantile_iterations=100,
                max_splits=4,
            )
        )
        self.assertEqual(
            set(result["methods"]),
            {"point_pto", "global_cqr_pto", "rcc", "oracle"},
        )
        self.assertEqual(result["methods"]["oracle"]["status"], "selected")
        summary = summarize_continuous_runs([result])
        self.assertEqual(summary["replications"], 1)
        self.assertEqual(summary["rcc_vs_global_cqr"]["comparable_runs"], 1)


if __name__ == "__main__":
    unittest.main()
