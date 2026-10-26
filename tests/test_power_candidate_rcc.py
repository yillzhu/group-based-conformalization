import unittest

import numpy as np

from conformal_cl.power.candidate_rcc import run_candidate_rcc


class TestCandidateRCC(unittest.TestCase):
    def test_selects_feasible_low_cost_candidate(self):
        rng = np.random.default_rng(1)
        design_x = rng.uniform(0., 1., size=(500, 2))
        calibration_x = rng.uniform(0., 1., size=(500, 2))
        design_scores = rng.normal(0., .01, size=500)
        calibration_scores = rng.normal(0., .01, size=500)
        candidate_x = rng.uniform(0., 1., size=(2_000, 2))
        prediction = candidate_x[:, 0] + candidate_x[:, 1] - .8
        cost = candidate_x.sum(axis=1)
        result = run_candidate_rcc(
            design_x, design_scores, calibration_x, calibration_scores,
            candidate_x, prediction, cost, max_splits=8,
        )
        self.assertEqual(result.status, "selected")
        self.assertLessEqual(prediction[result.index] + result.correction, 0.)


if __name__ == "__main__":
    unittest.main()
