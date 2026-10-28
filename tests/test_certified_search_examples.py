"""Model checks for the independently audited quadratic and tuition variants."""

from pathlib import Path
import runpy
import unittest

import numpy as np
from scipy.stats import norm

from conformal_cl.certified_search import SearchConfig


EXAMPLES = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "experiments/certified_search/run_other_examples.py")
)


class CertifiedExampleTests(unittest.TestCase):
    def test_quadratic_mesh_sizes(self):
        self.assertEqual(EXAMPLES["quadratic_grid"](1).shape, (801, 1))
        self.assertEqual(EXAMPLES["quadratic_grid"](2).shape, (81**2, 2))
        self.assertEqual(EXAMPLES["quadratic_grid"](3).shape, (31**3, 3))

    def test_quadratic_noise_scale_matches_population_variance(self):
        _, constraint_mean, sigma = EXAMPLES["quadratic_instance"](7, 2, .2)
        x = np.random.default_rng(4).uniform(-1, 1, (200_000, 2))
        self.assertAlmostEqual(np.std(constraint_mean(x)) * .2, sigma, delta=.01 * sigma)

    def test_quadratic_exact_risk_matches_simulation(self):
        _, constraint_mean, sigma = EXAMPLES["quadratic_instance"](2, 2, .2)
        x = np.array([[.2, -.4]])
        analytic = float(norm.cdf(constraint_mean(x)[0] / sigma))
        responses = constraint_mean(x)[0] + sigma * np.random.default_rng(8).normal(size=100_000)
        self.assertAlmostEqual(np.mean(responses > 0), analytic, delta=.005)

    def test_tuition_allocates_familywise_budget_across_segments(self):
        result = EXAMPLES["run_tuition"](
            0, SearchConfig(rounds=1, batch_size=100), 200, .4, .5
        )
        self.assertEqual(len(result["segments"]), 12)
        self.assertTrue(all(
            record["delta"] == .05 / 12 for record in result["segments"]
            if record["status"] == "mesh_feasible"
        ))


if __name__ == "__main__":
    unittest.main()
