import unittest

import numpy as np

from conformal_cl.transparent.capacity import (
    CASES, CapacityCase, empirical_comparison, population_comparison,
    population_width, solve_rcc, tolerance_rank,
)


class TestTransparentCapacity(unittest.TestCase):
    def test_analytic_mixture_quantile(self):
        self.assertAlmostEqual(population_width(CASES[0], 0., 4.), .09)
        self.assertAlmostEqual(population_width(CASES[1], 0., 4.), .4 * (1. - .1 / .7))
        for case in CASES:
            width = population_width(case, 0., 4.)
            weights = np.diff(case.edges) / 4.
            cdf = np.dot(weights, np.minimum(width / np.asarray(case.scales), 1.))
            self.assertAlmostEqual(cdf, .9)

    def test_exact_risk_matches_independent_simulation(self):
        rng = np.random.default_rng(7)
        for case in CASES:
            for x in (1.08, 1.2, 1.35):
                response = 1. - x + case.scale(x) * rng.uniform(-1., 1., 100000)
                self.assertAlmostEqual(np.mean(response > 0.), case.risk(x), delta=.005)

    def test_population_geometry_has_positive_and_negative_controls(self):
        homogeneous, aligned, interior = [population_comparison(c)["methods"] for c in CASES]
        self.assertAlmostEqual(homogeneous["rcc"]["capacity_cost"], 1.09)
        self.assertAlmostEqual(aligned["rcc"]["capacity_cost"], 1.09)
        self.assertLess(aligned["rcc"]["capacity_cost"], aligned["global_cqr"]["capacity_cost"])
        self.assertGreater(interior["rcc"]["capacity_cost"], 1.2)
        self.assertAlmostEqual(interior["known_regions_cqr"]["capacity_cost"], 1.09)
        # This quiet interval has mass .125 > delta=.1. Its failure at delta=.1
        # therefore reflects the regions produced by RCC, not an inherently
        # ineligible quiet interval. Smaller mass thresholds resolve it.
        self.assertAlmostEqual(solve_rcc(CASES[2], delta=.05)["capacity_cost"], 1.09)

    def test_empirical_rcc_retains_original_unresolved_fallback(self):
        case = CASES[1]
        rng = np.random.default_rng(8)
        data = case.draw(1000, rng)
        result = solve_rcc(case, data=data, max_splits=0)
        self.assertEqual(result["status"], "selected")
        self.assertEqual(result["route"], "unresolved")

    def test_one_dimensional_reduction_matches_existing_continuous_rcc(self):
        from conformal_cl.workforce.continuous_rcc import run_continuous_rcc
        from conformal_cl.workforce.models import LinearModel

        model = LinearModel(np.array([1., -1., 0.]), np.zeros(2), np.ones(2))
        for case in CASES:
            x, residual = case.draw(2000, np.random.default_rng(16))
            # A negligible, unused second coordinate lets the existing exact
            # polygon solver verify this continuous one-dimensional reduction.
            features = np.column_stack((x, np.full(len(x), .5e-9)))
            original = run_continuous_rcc(
                model, model, features, 1. - x + residual,
                domain_lower=np.array([0., 0.]),
                domain_upper=np.array([4., 1e-9]),
                objective_slope=np.array([1., 0.]),
            )
            reduced = solve_rcc(case, data=(x, residual))
            self.assertEqual(original.status, reduced["status"])
            self.assertAlmostEqual(original.objective, reduced["capacity_cost"], places=7)

    def test_tolerance_rank_has_simultaneous_failure_budget(self):
        from scipy.stats import binom
        rank = tolerance_rank(200, alpha=.1, beta=.05 / 3.)
        self.assertLessEqual(binom.sf(rank - 1, 200, .9), .05 / 3.)
        self.assertGreater(binom.sf(rank - 2, 200, .9), .05 / 3.)
        self.assertIsNone(tolerance_rank(1, alpha=.1, beta=.05))

    def test_correctly_scaled_model_removes_population_localization_gain(self):
        for case in CASES:
            methods = population_comparison(case)["methods"]
            self.assertAlmostEqual(methods["known_scale_cqr"]["capacity_cost"], 1.09)
            self.assertAlmostEqual(methods["chance_oracle"]["capacity_cost"], 1.08)

    def test_invalid_scale_definition_is_rejected(self):
        with self.assertRaises(ValueError):
            CapacityCase("invalid", (0., 1., .5), (.1, .2))


if __name__ == "__main__":
    unittest.main()
