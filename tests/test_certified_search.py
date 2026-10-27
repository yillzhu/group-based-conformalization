"""Checks for the independent pointwise audit, not an RCC guarantee."""

import numpy as np
import unittest

from conformal_cl.certified_search import (
    SearchConfig,
    binomial_upper_bound,
    run_certified_search,
)


class CertifiedSearchTests(unittest.TestCase):
    def test_clopper_pearson_bound_special_cases(self):
        self.assertAlmostEqual(binomial_upper_bound(0, 100, 0.05), 1 - 0.05**(1 / 100))
        self.assertEqual(binomial_upper_bound(100, 100, 0.05), 1)
        with self.assertRaises(ValueError):
            binomial_upper_bound(101, 100, 0.05)
        with self.assertRaises(ValueError):
            SearchConfig(capacity_index=1.5)


    def test_search_certifies_only_fresh_audited_decisions(self):
        candidates = np.arange(10, dtype=float)[:, None]
        seen = []

        def simulator(decision, size, rng):
            seen.append((decision.copy(), size))
            return np.full(size, 1.0 if decision[0] < 5 else -1.0)

        result = run_certified_search(
            candidates=candidates, costs=candidates[:, 0],
            initial_x=np.repeat(candidates, 25, axis=0),
            initial_response=np.repeat(np.where(candidates[:, 0] < 5, 1., -1.), 25),
            simulator=simulator, rng=np.random.default_rng(7),
            config=SearchConfig(rounds=2, batch_size=100),
        )
        self.assertEqual(len(seen), len(result["audits"]))
        self.assertEqual(len(seen), 2)
        self.assertTrue(all(size == 100 for _, size in seen))
        self.assertIsNotNone(result["selected"])
        self.assertGreaterEqual(result["selected"]["decision"][0], 5)
        self.assertLessEqual(result["selected"]["risk_upper_bound"], 0.1)


    def test_all_failed_audits_return_no_decision(self):
        x = np.linspace(0, 1, 100)[:, None]
        result = run_certified_search(
            candidates=x, costs=x[:, 0], initial_x=x,
            initial_response=np.ones(len(x)),
            simulator=lambda decision, size, rng: np.ones(size),
            rng=np.random.default_rng(1),
            config=SearchConfig(rounds=2, batch_size=50),
        )
        self.assertIsNone(result["selected"])
        self.assertFalse(any(audit["accepted"] for audit in result["audits"]))

    def test_known_capacity_shift_learns_requirement(self):
        dedicated = np.repeat(np.array([0., 1.]), 50)
        flexible = np.tile(np.linspace(0., 9., 50), 2)
        train_x = np.column_stack((dedicated, flexible))
        candidates = np.column_stack((np.zeros(10), np.arange(10.)))

        def simulator(decision, size, rng):
            return np.full(size, 5. - decision[1])

        result = run_certified_search(
            candidates=candidates, costs=candidates[:, 1],
            initial_x=train_x, initial_response=5. - flexible,
            simulator=simulator, rng=np.random.default_rng(4),
            config=SearchConfig(rounds=1, batch_size=100, capacity_index=1),
        )
        self.assertEqual(result["selected"]["decision"], [0., 5.])
        self.assertAlmostEqual(result["audits"][0]["predicted_quantile"], 0.)


if __name__ == "__main__":
    unittest.main()
