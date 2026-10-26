import unittest

import numpy as np

from conformal_cl.order_stats import conformal_order_statistic
from conformal_cl.transparent.capacity import CASES
from conformal_cl.transparent.learned import model_diagnostics, solve_learned_rcc
from conformal_cl.transparent.quantile_models import QuantileSpline, fit_quantiles


class TestTransparentLearned(unittest.TestCase):
    def test_exact_spline_optimization_handles_nonmonotone_prediction(self):
        knots = np.arange(5, dtype=float)
        upper = np.array([1., .5, -.2, .4, -1.])
        model = QuantileSpline(knots, upper - .5, upper, np.array([1., -1.]))
        self.assertAlmostEqual(model.cheapest_feasible(0., 4.), 1. + .5 / .7)
        self.assertIsNone(model.cheapest_feasible(0., 1.))
        self.assertEqual(model.extrema(0., 4., 1), (-1., 4., 1., 0.))

    def test_oracle_model_has_zero_excess_loss_and_correct_coverage(self):
        knots = np.array([0., 4.])
        mean = 1. - knots
        model = QuantileSpline(knots, mean - .09, mean + .09, np.array([1., -1.]))
        result = model_diagnostics(CASES[0], model, np.linspace(0., 4., 200))
        self.assertAlmostEqual(result["relative_excess_pinball"], 0.)
        self.assertAlmostEqual(result["raw_upper_coverage"], .95)
        self.assertAlmostEqual(result["raw_interval_coverage"], .9)

    def test_piecewise_model_is_continuous_and_noncrossing(self):
        rng = np.random.default_rng(31)
        x, residual = CASES[1].draw(2000, rng)
        vx, ve = CASES[1].draw(1000, rng)
        model, metadata = fit_quantiles(x, 1. - x + residual, vx, 1. - vx + ve)
        predictions = model.predict(np.r_[model.knots, np.linspace(0., 4., 2001)])
        self.assertTrue(np.all(predictions[:, 0] <= predictions[:, 1]))
        for point in model.knots[1:-1]:
            difference = model.predict([point - 1e-12, point + 1e-12])
            np.testing.assert_allclose(difference[0], difference[1], atol=1e-7)
        self.assertLessEqual(metadata["validation_pinball"],
                             min(c["validation_pinball"] for c in metadata["candidates"]))

    def test_reduced_solver_matches_existing_continuous_rcc(self):
        from conformal_cl.workforce.continuous_rcc import run_continuous_rcc

        for index, case in enumerate(CASES):
            rng = np.random.default_rng(10 + index)
            x, e = case.draw(1500, rng)
            vx, ve = case.draw(1000, rng)
            cx, ce = case.draw(2000, rng)
            model, _ = fit_quantiles(x, 1. - x + e, vx, 1. - vx + ve)
            y = 1. - cx + ce
            original = run_continuous_rcc(
                model.as_workforce_model(0), model.as_workforce_model(1),
                np.column_stack((cx, np.full(len(cx), .5e-9))), y,
                domain_lower=np.array([0., 0.]), domain_upper=np.array([4., 1e-9]),
                objective_slope=np.array([1., 0.]),
            )
            decision, _ = solve_learned_rcc(model, cx, y)
            self.assertEqual(original.status, "selected")
            self.assertAlmostEqual(original.objective, decision, places=6)

            predicted = model.predict(cx)
            scores = np.maximum(y - predicted[:, 1], predicted[:, 0] - y)
            correction = conformal_order_statistic(scores, alpha=.1, level_adjustment=1. / len(cx))
            no_splits, _ = solve_learned_rcc(model, cx, y, max_splits=0)
            self.assertAlmostEqual(no_splits, model.cheapest_feasible(0., 4., correction))


if __name__ == "__main__":
    unittest.main()
