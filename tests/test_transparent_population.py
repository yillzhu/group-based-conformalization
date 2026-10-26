import unittest

import numpy as np
from scipy.integrate import quad

from conformal_cl.transparent.capacity import CASES
from conformal_cl.transparent.population import PopulationScoreLaw, correction_decomposition
from conformal_cl.transparent.quantile_models import QuantileSpline


class TestPopulationScores(unittest.TestCase):
    def test_exact_quantiles_and_negative_corrections(self):
        knots = np.array([0., 4.])
        for width in (0., .04, .09, .2):
            mean = 1. - knots
            model = QuantileSpline(knots, mean - width, mean + width, np.array([1., -1.]))
            law = PopulationScoreLaw(CASES[0], model, 0., 4.)
            self.assertAlmostEqual(law.quantile(), .09 - width)
            self.assertAlmostEqual(law.cdf(-width - .01), 0.)
            self.assertAlmostEqual(law.cdf(.09 - width), .9)
            self.assertAlmostEqual(law.cdf(.2), 1.)

    def test_piecewise_integral_matches_independent_quadrature(self):
        knots = np.array([0., .7, 1.1, 1.7, 4.])
        mean = 1. - knots
        model = QuantileSpline(knots, mean - np.array([.2, .05, .3, .1, .6]),
                               mean + np.array([.1, .3, .02, .7, .15]), np.array([1., -1.]))
        case = CASES[2]
        law = PopulationScoreLaw(case, model, .2, 3.8)
        for q in (-.2, -.04, 0., .13, .5):
            def integrand(x):
                lower, upper = model.predict([x])[0]
                scale, m = float(case.scale(x)), 1. - x
                return max(0., np.clip((upper + q - m + scale) / (2. * scale), 0., 1.)
                           - np.clip((lower - q - m + scale) / (2. * scale), 0., 1.))
            value = quad(integrand, .2, 3.8,
                         points=sorted(set([.7, .9, 1.1, 1.4, 1.7])),
                         epsabs=1e-9, limit=300)[0] / 3.6
            self.assertAlmostEqual(law.cdf(q), value, places=7)

    def test_decomposition_is_an_identity_not_an_unbiasedness_claim(self):
        knots = np.array([0., 4.])
        model = QuantileSpline(knots, 1. - knots, 1. - knots, np.array([1., -1.]))
        result = correction_decomposition(CASES[1], model, .35,
                                           {"interval": [0., 1.1], "correction": .08})
        self.assertAlmostEqual(result["empirical_local_minus_global"],
                               result["population_local_minus_global"]
                               + result["sampling_and_selection_remainder"])
        self.assertAlmostEqual(result["local_population_correction"], .09)

    def test_uniform_endpoint_error_brackets_every_population_correction(self):
        knots = np.linspace(0., 4., 21)
        mean = 1. - knots
        epsilon = .005
        model = QuantileSpline(knots, mean - .09 + epsilon * np.sin(4. * knots),
                               mean + .09 + epsilon * np.cos(3. * knots), np.array([1., -1.]))
        for left, right in ((0., 4.), (0., .1), (.3, 1.2), (1., 1.1), (2.3, 4.)):
            q = PopulationScoreLaw(CASES[0], model, left, right).quantile()
            self.assertLessEqual(abs(q), epsilon + 1e-12)


if __name__ == "__main__":
    unittest.main()
