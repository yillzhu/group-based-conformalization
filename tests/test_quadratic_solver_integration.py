import os
import unittest


@unittest.skipUnless(
    os.environ.get("RUN_GUROBI_INTEGRATION") == "1",
    "set RUN_GUROBI_INTEGRATION=1 in the licensed quadratic environment",
)
class QuadraticSolverIntegrationTests(unittest.TestCase):
    def test_end_to_end_smoke_case_has_structured_statuses(self):
        import numpy as np

        from conformal_cl.quadratic.benchmark import run_one

        result = run_one(seed=0, dim=1, noise_std=0.1, P=2, alpha=0.1)
        for index in (0, 5, 10, 15, 20):
            self.assertTrue(np.isfinite(np.asarray(result[index], dtype=float)).all())
        self.assertEqual(tuple(result[index] for index in (1, 6, 11, 16)), (1, 1, 1, True))
        self.assertEqual(result[22:], ("Optimal",) * 5)


if __name__ == "__main__":
    unittest.main()
