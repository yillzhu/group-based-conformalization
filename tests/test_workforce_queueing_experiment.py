import unittest

from conformal_cl.workforce.queueing import QueueConfig
from conformal_cl.workforce.queueing_experiment import (
    QueueExperimentConfig,
    estimate_queue_oracle,
    run_queue_experiment,
    summarize_queue_runs,
)


class QueueingExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oracle = estimate_queue_oracle(30, seed=44)

    def test_smoke_run_returns_all_methods(self):
        result = run_queue_experiment(
            QueueExperimentConfig(
                seed=3,
                training_size=80,
                design_size=300,
                calibration_size=500,
                max_splits=2,
                min_design_size=20,
                quantile_iterations=100,
            ),
            self.oracle,
        )
        self.assertEqual(
            set(result["methods"]),
            {
                "point_pto",
                "global_conformal_pto",
                "global_cqr_pto",
                "refit_global_conformal_pto",
                "domain_conformal_pto",
                "group_based",
                "oblique_group_based",
                "rcc_group_based",
                "risk_stratified",
                "oracle",
            },
        )
        self.assertGreaterEqual(len(result["model"]["partition"]), 1)
        rcc = result["methods"]["rcc_group_based"]
        if rcc["status"] == "selected":
            self.assertIn("selected_cell", rcc)
            self.assertGreaterEqual(rcc["selected_cell"]["candidate_count"], 1)

    def test_oracle_configuration_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            run_queue_experiment(
                QueueExperimentConfig.smoke(),
                self.oracle,
                queue_config=QueueConfig(service_level_target=0.75),
            )

    def test_summary_distinguishes_no_solution_and_unsafe_selection(self):
        run = run_queue_experiment(
            QueueExperimentConfig(
                seed=5,
                training_size=80,
                design_size=300,
                calibration_size=500,
                max_splits=2,
                min_design_size=20,
                quantile_iterations=100,
            ),
            self.oracle,
        )
        summary = summarize_queue_runs([run])
        self.assertEqual(summary["replications"], 1)
        self.assertIn("unsafe_selection_count", summary["methods"]["point_pto"])


if __name__ == "__main__":
    unittest.main()
