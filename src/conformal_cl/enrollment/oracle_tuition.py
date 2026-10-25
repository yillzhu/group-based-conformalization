"""Synthetic tuition-response benchmark with a known monotone oracle.

Unlike the recovered JANOS files, this generator publishes the response mechanism.
Each observation is the enrollment rate of a homogeneous applicant segment offered
one merit level.  Repeated cohorts make the conformal target observable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def segment_table(n_segments=12):
    """Return deterministic applicant segments and their cohort sizes."""

    if n_segments < 3:
        raise ValueError("n_segments must be at least 3")
    index = np.arange(n_segments)
    sat = np.linspace(1050.0, 1550.0, n_segments)
    gpa = 2.55 + 1.25 * (index / (n_segments - 1)) ** 0.85
    batch_pattern = np.array([24, 60, 180])
    batch_size = batch_pattern[index % len(batch_pattern)]
    return pd.DataFrame(
        {
            "segment": index,
            "SAT": sat,
            "GPA": gpa,
            "batch_size": batch_size,
        }
    )


def oracle_probability(sat, gpa, merit):
    """Known enrollment probability, monotone increasing in merit.

    Merit is measured in thousands of dollars. Higher-academic-profile segments
    have a lower baseline yield and a slightly larger scholarship response.
    """

    sat = np.asarray(sat, dtype=float)
    gpa = np.asarray(gpa, dtype=float)
    merit = np.asarray(merit, dtype=float)
    if np.any((merit < 0.0) | (merit > 25.0)):
        raise ValueError("merit must lie in [0, 25]")

    sat_scaled = (sat - 1300.0) / 250.0
    gpa_scaled = (gpa - 3.2) / 0.65
    merit_slope = 0.145 + 0.025 / (1.0 + np.exp(-sat_scaled))
    logit = (
        -1.35
        - 0.70 * sat_scaled
        - 0.45 * gpa_scaled
        + merit_slope * merit
        + 0.35 * np.tanh((merit - 8.0) / 4.0)
    )
    return 1.0 / (1.0 + np.exp(-logit))


def generate_cohort_rates(n_rows, *, seed, n_segments=12):
    """Generate randomized merit offers and observed cohort enrollment rates."""

    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    rng = np.random.default_rng(seed)
    segments = segment_table(n_segments)
    chosen = rng.integers(0, n_segments, size=n_rows)
    merit = rng.uniform(0.0, 25.0, size=n_rows)
    rows = segments.iloc[chosen].reset_index(drop=True)
    probability = oracle_probability(rows["SAT"], rows["GPA"], merit)
    enrolled = rng.binomial(rows["batch_size"].to_numpy(), probability)
    return pd.DataFrame(
        {
            "segment": rows["segment"].to_numpy(),
            "SAT": rows["SAT"].to_numpy(),
            "GPA": rows["GPA"].to_numpy(),
            "merit": merit,
            "batch_size": rows["batch_size"].to_numpy(),
            "enrolled": enrolled,
            "enrollment_rate": enrolled / rows["batch_size"].to_numpy(),
            "oracle_probability": probability,
        }
    )


def project_merit_monotone_(model, merit_index=2):
    """Project a one-hidden-layer ReLU network to a sufficient monotone cone.

    The expected architecture is ``Linear -> ReLU -> Linear`` with an optional
    final sigmoid. Nonnegative merit input weights and nonnegative output weights
    make every output nondecreasing in merit.
    """

    first = model[0]
    output = model[2]
    with torch.no_grad():
        first.weight[:, merit_index].clamp_(min=0.0)
        output.weight.clamp_(min=0.0)


def monotone_weights_valid(model, merit_index=2, tolerance=0.0):
    """Check the sufficient weight conditions used by the projection."""

    first_merit = model[0].weight[:, merit_index].detach().cpu().numpy()
    output = model[2].weight.detach().cpu().numpy()
    return bool(np.all(first_merit >= -tolerance) and np.all(output >= -tolerance))
