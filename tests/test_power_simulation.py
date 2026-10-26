import unittest

import numpy as np

from conformal_cl.power import (
    DCSecuritySimulator,
    load_case39,
    sample_dispatches,
    sample_security_focused_dispatches,
)


class TestPowerSimulation(unittest.TestCase):
    def test_case_data_and_connected_outages(self):
        case = load_case39()
        self.assertAlmostEqual(case.demand, 6254.23)
        self.assertEqual(len(case.p_max), 10)
        simulator = DCSecuritySimulator(case)
        self.assertGreater(simulator.contingencies, 30)
        self.assertLess(simulator.contingencies, len(case.branch_x))
        self.assertAlmostEqual(simulator.outage_probability, 0.05)

    def test_outage_probability_is_validated(self):
        with self.assertRaises(ValueError):
            DCSecuritySimulator(load_case39(), outage_probability=1.01)

    def test_dispatch_sampler_respects_polytope(self):
        case = load_case39()
        samples = sample_dispatches(case, 50, np.random.default_rng(2), burn_in=20)
        np.testing.assert_allclose(samples.sum(axis=1), case.demand, atol=1e-8)
        self.assertTrue(np.all(samples >= case.p_min - 1e-8))
        self.assertTrue(np.all(samples <= case.p_max + 1e-8))
        self.assertGreater(np.std(samples[:, 0]), 1.)

    def test_response_is_finite_and_reproducible(self):
        case = load_case39()
        simulator = DCSecuritySimulator(case)
        p = sample_dispatches(case, 12, np.random.default_rng(3), burn_in=20)
        first = simulator.sample_response(p, np.random.default_rng(4))
        second = simulator.sample_response(p, np.random.default_rng(4))
        np.testing.assert_allclose(first, second)
        self.assertTrue(np.isfinite(first).all())
        self.assertEqual(first.shape, (12,))

    def test_security_focused_sampler_respects_nominal_limits(self):
        case = load_case39()
        simulator = DCSecuritySimulator(case)
        samples = sample_security_focused_dispatches(
            simulator, 30, np.random.default_rng(7), burn_in=20
        )
        injection = samples @ simulator.generator_map.T - case.loads
        flow = injection @ simulator.nominal_flow_matrix.T
        self.assertTrue(np.all(np.abs(flow) <= case.branch_rate + 1e-7))
        self.assertTrue(np.all(samples >= .02 * case.p_max - 1e-7))
        self.assertTrue(np.all(samples <= .98 * case.p_max + 1e-7))


if __name__ == "__main__":
    unittest.main()
