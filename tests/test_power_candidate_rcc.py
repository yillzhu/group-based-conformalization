import unittest

import numpy as np

from conformal_cl.power.candidate_rcc import run_candidate_rcc


class TestCandidateRCC(unittest.TestCase):
    def test_selects_low_cost_candidate_when_entire_region_is_safe(self):
        rng = np.random.default_rng(1)
        design_x = rng.uniform(0., 1., size=(500, 2))
        calibration_x = rng.uniform(0., 1., size=(500, 2))
        design_scores = rng.normal(0., .01, size=500)
        calibration_scores = rng.normal(0., .01, size=500)
        candidate_x = rng.uniform(0., 1., size=(2_000, 2))
        prediction = -.2 - .1 * candidate_x[:, 0]
        cost = candidate_x.sum(axis=1)
        result = run_candidate_rcc(
            design_x, design_scores, calibration_x, calibration_scores,
            candidate_x, prediction, cost, max_splits=0,
        )
        self.assertEqual(result.status, "selected")
        self.assertLessEqual(prediction[result.index] + result.correction, 0.)
        self.assertEqual(result.certified_safe_regions, 1)

    def test_rejects_individual_points_in_uncertified_region(self):
        rng = np.random.default_rng(2)
        design_x = rng.uniform(0., 1., size=(500, 2))
        calibration_x = rng.uniform(0., 1., size=(500, 2))
        scores = np.zeros(500)
        candidate_x = np.array([[.1, .1], [.9, .9]])
        prediction = np.array([-.1, .1])
        result = run_candidate_rcc(
            design_x, scores, calibration_x, scores, candidate_x,
            prediction, candidate_x.sum(axis=1), max_splits=0,
        )
        self.assertEqual(result.status, "no_safe_region")
        self.assertEqual(result.certified_safe_regions, 0)
        self.assertEqual(result.uncertified_regions, 1)


if __name__ == "__main__":
    unittest.main()
