"""Validation and aggregation for corrected quadratic result CSVs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLUMNS = ["seed", "dim", "noise_std", "P", "alpha"]
METHODS = {
    "abs": "Standard",
    "cqr": "CQR",
    "wrapper": "Wrapper",
    "rcc": "RCC",
}


def parse_vector(value, dimension):
    """Parse the legacy NumPy-style vector representation, or return ``None``."""

    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        return None
    values = np.fromstring(text.strip("[]"), sep=" ")
    if values.size != dimension or not np.isfinite(values).all():
        raise ValueError(f"invalid dimension-{dimension} vector: {value!r}")
    return values


def parse_feasibility_flag(value):
    """Normalize legacy feasibility flags while retaining missing-solution state."""

    text = str(value).strip().lower().strip("[]")
    if text in {"true", "1", "1.0"}:
        return 1
    if text in {"false", "0", "0.0"}:
        return 0
    if text in {"-1", "-1.0", "2", "2.0"}:
        return -1
    raise ValueError(f"unknown feasibility flag: {value!r}")


def load_corrected_results(paths):
    """Load, validate, and normalize one or more nonoverlapping result batches."""

    frames = [pd.read_csv(path) for path in map(Path, paths)]
    if not frames:
        raise ValueError("at least one result CSV is required")
    results = pd.concat(frames, ignore_index=True)
    required = set(KEY_COLUMNS + ["opt_true", "t_true", "status_true"])
    for suffix in METHODS:
        required.update(
            [f"opt_{suffix}", f"f_{suffix}", f"t_{suffix}", f"status_{suffix}"]
        )
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"missing result columns: {sorted(missing)}")
    if results.duplicated(KEY_COLUMNS).any():
        raise ValueError("duplicate experimental parameter keys")

    for column in KEY_COLUMNS:
        results[column] = pd.to_numeric(results[column], errors="raise")
    if not np.isfinite(results[KEY_COLUMNS].to_numpy()).all():
        raise ValueError("parameter keys must be finite")

    results["true_returned"] = [
        parse_vector(value, int(dim)) is not None
        for value, dim in zip(results["opt_true"], results["dim"])
    ]
    if not (
        results["true_returned"]
        == results["status_true"].astype(str).eq("Optimal")
    ).all():
        raise ValueError("true solution vectors and structured statuses disagree")
    true_times = pd.to_numeric(results["t_true"], errors="raise")
    if not np.isfinite(true_times).all() or (true_times < 0).any():
        raise ValueError("true solve: invalid recorded solver time")
    results["t_true"] = true_times

    for suffix in METHODS:
        flags = results[f"f_{suffix}"].map(parse_feasibility_flag)
        returned = pd.Series(
            [
                parse_vector(value, int(dim)) is not None
                for value, dim in zip(results[f"opt_{suffix}"], results["dim"])
            ],
            index=results.index,
        )
        statuses = results[f"status_{suffix}"].astype(str)
        if not (returned == statuses.str.startswith("Optimal")).all():
            raise ValueError(f"{suffix}: vectors and structured statuses disagree")
        if not (returned == flags.isin([0, 1])).all():
            raise ValueError(f"{suffix}: vectors and feasibility flags disagree")
        times = pd.to_numeric(results[f"t_{suffix}"], errors="raise")
        if not np.isfinite(times).all() or (times < 0).any():
            raise ValueError(f"{suffix}: invalid recorded solver time")
        results[f"flag_{suffix}"] = flags
        results[f"returned_{suffix}"] = returned
        results[f"t_{suffix}"] = times
    return results


def summarize_corrected_results(results):
    """Return corrected feasibility and recorded-time summary tables."""

    true_feasible = results.loc[results["true_returned"]]
    feasibility_rows = []
    timing_rows = []
    for suffix, label in METHODS.items():
        flags = true_feasible[f"flag_{suffix}"]
        n_instances = len(flags)
        n_returned = int(flags.isin([0, 1]).sum())
        n_feasible = int(flags.eq(1).sum())
        feasibility_rows.append(
            {
                "Method": label,
                "N_True_Feasible_Instances": n_instances,
                "N_Returned": n_returned,
                "N_True_Feasible_Solutions": n_feasible,
                "Rate_All_True_Feasible": (
                    n_feasible / n_instances if n_instances else np.nan
                ),
                "Rate_Returned_Solutions": (
                    n_feasible / n_returned if n_returned else np.nan
                ),
            }
        )
        for dimension, group in true_feasible.groupby("dim"):
            timing_rows.append(
                {
                    "dim": int(dimension),
                    "Method": label,
                    "N": len(group),
                    "Mean_Recorded_Seconds": float(group[f"t_{suffix}"].mean()),
                }
            )
    return pd.DataFrame(feasibility_rows), pd.DataFrame(timing_rows)
