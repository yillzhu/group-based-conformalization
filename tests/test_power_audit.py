import unittest

from conformal_cl.power.audit import PowerAuditConfig, run_power_audit


class TestPowerAudit(unittest.TestCase):
    def test_tiny_audit_has_independent_partition_and_calibration(self):
        config = PowerAuditConfig(
            seed=5,
            training_size=500,
            design_size=300,
            calibration_size=300,
            candidate_size=1_000,
            evaluation_size=500,
            model_iterations=10,
            model_leaf_size=10,
            partition_leaves=2,
            partition_leaf_size=50,
        )
        result = run_power_audit(config)
        self.assertEqual(result["case"]["generators"], 10)
        self.assertEqual(len(result["partition"]["leaves"]), 2)
        self.assertIn("global_conformal", result["methods"])
        self.assertIn("local_diagnostic", result["methods"])


if __name__ == "__main__":
    unittest.main()
