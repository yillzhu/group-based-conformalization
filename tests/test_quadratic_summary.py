import unittest

import pandas as pd

from conformal_cl.quadratic.summary import (
    load_corrected_results,
    parse_feasibility_flag,
    parse_vector,
    summarize_corrected_results,
)


class QuadraticSummaryTests(unittest.TestCase):
    def test_legacy_value_parsing(self):
        self.assertEqual(parse_vector("[0.1 -0.2]", 2).tolist(), [0.1, -0.2])
        self.assertIsNone(parse_vector(float("nan"), 1))
        self.assertEqual(parse_feasibility_flag("True"), 1)
        self.assertEqual(parse_feasibility_flag("[2]"), -1)

    def test_structured_status_is_required_to_match_vector(self):
        frame = pd.DataFrame(
            [
                {
                    "seed": 0,
                    "dim": 1,
                    "noise_std": 0.1,
                    "P": 2,
                    "alpha": 0.1,
                    "opt_true": "[0.]",
                    "t_true": 0.1,
                    "status_true": "Optimal",
                    **{
                        field: value
                        for suffix in ("abs", "cqr", "wrapper", "rcc")
                        for field, value in (
                            (f"opt_{suffix}", "[0.]"),
                            (f"f_{suffix}", 1),
                            (f"t_{suffix}", 0.1),
                            (f"status_{suffix}", "Infeasible"),
                        )
                    },
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "vectors and structured statuses"):
            load_corrected_results_from_frame(frame)

    def test_summary_uses_all_true_feasible_denominator(self):
        rows = []
        for seed, flag in [(0, 1), (1, -1)]:
            row = {"seed": seed, "dim": 1, "true_returned": True}
            for suffix in ("abs", "cqr", "wrapper", "rcc"):
                row[f"flag_{suffix}"] = flag
                row[f"t_{suffix}"] = 0.5
            rows.append(row)
        table, _ = summarize_corrected_results(pd.DataFrame(rows))
        self.assertEqual(table.loc[0, "N_True_Feasible_Instances"], 2)
        self.assertEqual(table.loc[0, "N_Returned"], 1)
        self.assertEqual(table.loc[0, "Rate_All_True_Feasible"], 0.5)

    def test_true_solver_time_must_be_finite_and_nonnegative(self):
        frame = valid_result_frame()
        frame.loc[0, "t_true"] = -0.1
        with self.assertRaisesRegex(ValueError, "true solve"):
            load_corrected_results_from_frame(frame)


def load_corrected_results_from_frame(frame):
    """Exercise the file loader without adding a public frame-only API."""

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "results.csv"
        frame.to_csv(path, index=False)
        return load_corrected_results([path])


def valid_result_frame():
    row = {
        "seed": 0,
        "dim": 1,
        "noise_std": 0.1,
        "P": 2,
        "alpha": 0.1,
        "opt_true": "[0.]",
        "t_true": 0.1,
        "status_true": "Optimal",
    }
    for suffix in ("abs", "cqr", "wrapper", "rcc"):
        row.update(
            {
                f"opt_{suffix}": "[0.]",
                f"f_{suffix}": 1,
                f"t_{suffix}": 0.1,
                f"status_{suffix}": "Optimal",
            }
        )
    return pd.DataFrame([row])


if __name__ == "__main__":
    unittest.main()
