import unittest

import numpy as np

from conformal_cl.workforce.conformal import (
    calibrate_partition,
    calibrate_prediction_strata,
    design_oblique_partition,
    design_partition,
)
from conformal_cl.workforce.experiment import ExperimentConfig, run_experiment
from conformal_cl.workforce.study import summarize_runs
from conformal_cl.workforce.problem import sample_responses, staffing_grid


class WorkforceTests(unittest.TestCase):
    def test_more_staff_never_increases_unmet_workload_for_same_scenario(self):
        seed = 13
        low = sample_responses(np.tile([8.0, 4.0], (200, 1)), np.random.default_rng(seed))
        high = sample_responses(np.tile([9.0, 5.0], (200, 1)), np.random.default_rng(seed))
        self.assertTrue(np.all(high <= low))

    def test_partition_uses_separate_final_calibration_scores(self):
        candidates = staffing_grid(range(4, 8), range(0, 5))
        rng = np.random.default_rng(4)
        design_x = candidates[rng.integers(0, len(candidates), size=500)]
        design_scores = design_x[:, 0] - 5.0
        upper = 4.0 - candidates[:, 0] - candidates[:, 1]
        cells = design_partition(
            design_x,
            design_scores,
            candidates,
            upper,
            alpha=0.1,
            max_splits=2,
            min_design_size=30,
        )
        cal_x = candidates[rng.integers(0, len(candidates), size=1000)]
        first = calibrate_partition(cells, cal_x, np.zeros(1000), alpha=0.1)
        second = calibrate_partition(cells, cal_x, np.ones(1000), alpha=0.1)
        self.assertTrue(np.all(first.corrections == 0.0))
        self.assertTrue(np.all(second.corrections == 1.0))
        self.assertEqual(int(first.calibration_counts.sum()), 1000)

    def test_raw_boundary_can_trigger_split_when_root_correction_is_too_large(self):
        candidates = staffing_grid(range(4, 8), range(0, 5))
        rng = np.random.default_rng(17)
        design_x = candidates[rng.integers(0, len(candidates), size=500)]
        # The root correction is positive enough to make every corrected upper
        # endpoint infeasible, while the raw fitted endpoint crosses zero.
        design_scores = np.where(design_x[:, 0] <= 5, 10.0, 5.0)
        raw_upper = candidates[:, 0] - 5.5
        cells = design_partition(
            design_x,
            design_scores,
            candidates,
            raw_upper,
            alpha=0.1,
            max_splits=1,
            min_design_size=30,
        )
        self.assertEqual(len(cells), 2)

    def test_prediction_strata_use_every_calibration_observation(self):
        candidate_prediction = np.arange(8, dtype=float)
        calibration_prediction = np.repeat(candidate_prediction, 20)
        scores = calibration_prediction / 10.0
        calibrated = calibrate_prediction_strata(
            candidate_prediction,
            calibration_prediction,
            scores,
            alpha=0.1,
            strata=4,
        )
        self.assertEqual(int(calibrated.calibration_counts.sum()), len(scores))
        self.assertEqual(len(calibrated.corrections), 4)

    def test_oblique_partition_covers_candidates_once(self):
        candidates = staffing_grid(range(4, 8), range(0, 5))
        rng = np.random.default_rng(19)
        design_x = candidates[rng.integers(0, len(candidates), size=800)]
        scores = design_x[:, 0] - design_x[:, 1]
        upper = 5.0 - candidates[:, 0] - candidates[:, 1]
        cells = design_oblique_partition(
            design_x,
            scores,
            candidates,
            upper,
            alpha=0.1,
            max_splits=3,
            min_design_size=40,
        )
        membership = sum(cell.contains(candidates).astype(int) for cell in cells)
        np.testing.assert_array_equal(membership, np.ones(len(candidates), dtype=int))

    def test_smoke_experiment_is_deterministic_and_returns_all_methods(self):
        config = ExperimentConfig.smoke(seed=2)
        first = run_experiment(config)
        second = run_experiment(config)
        self.assertEqual(first, second)
        self.assertEqual(
            set(first["methods"]),
            {"point_pto", "global_conformal_pto", "group_based", "oracle"},
        )

    def test_uniform_training_negative_control_runs(self):
        config = ExperimentConfig.smoke(seed=3)
        config = ExperimentConfig(**{**config.__dict__, "training_min_dedicated": 4})
        result = run_experiment(config)
        self.assertEqual(result["config"]["training_min_dedicated"], 4)

    def test_summary_reports_paired_cost_comparison(self):
        runs = [run_experiment(ExperimentConfig.smoke(seed=seed)) for seed in range(2)]
        summary = summarize_runs(runs)
        self.assertEqual(summary["replications"], 2)
        comparisons = summary["group_vs_global"]
        self.assertEqual(comparisons["comparable_runs"], 2)
        self.assertAlmostEqual(
            comparisons["group_cheaper_rate"]
            + comparisons["equal_cost_rate"]
            + comparisons["group_costlier_rate"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
