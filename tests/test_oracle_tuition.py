import unittest

import numpy as np
import torch

from conformal_cl.enrollment.models import (
    create_enrollment_net,
    create_quantile_net,
)
from conformal_cl.enrollment.oracle_tuition import (
    generate_cohort_rates,
    monotone_weights_valid,
    oracle_probability,
    project_merit_monotone_,
)


class OracleTuitionTests(unittest.TestCase):
    def test_oracle_is_monotone_in_merit(self):
        merit = np.linspace(0, 25, 501)
        for sat, gpa in [(1050, 2.55), (1300, 3.2), (1550, 3.8)]:
            probability = oracle_probability(sat, gpa, merit)
            self.assertTrue(np.all(np.diff(probability) > 0))

    def test_generation_is_reproducible_and_bounded(self):
        first = generate_cohort_rates(100, seed=7)
        second = generate_cohort_rates(100, seed=7)
        self.assertTrue(first.equals(second))
        self.assertTrue(first["enrollment_rate"].between(0, 1).all())
        self.assertTrue(first["merit"].between(0, 25).all())

    def test_projection_makes_supported_networks_monotone(self):
        for model in (create_enrollment_net(), create_quantile_net(hidden_size=10)[:3]):
            with torch.no_grad():
                model[0].weight[:, 2].fill_(-1)
                model[2].weight.fill_(-1)
            project_merit_monotone_(model)
            self.assertTrue(monotone_weights_valid(model))


if __name__ == "__main__":
    unittest.main()
