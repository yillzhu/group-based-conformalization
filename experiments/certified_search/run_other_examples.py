#!/usr/bin/env python3
"""Fresh-audit search on stochastic quadratics and synthetic tuition cohorts.

These are not reproductions of the historical deterministic quadratic or
observational JANOS tuition claims. Both experiments supply an explicit iid
simulator at each chosen decision, as required by pointwise certification.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from scipy.stats import binom, norm

from conformal_cl.certified_search import SearchConfig, run_certified_search
from conformal_cl.enrollment.oracle_tuition import (
    generate_cohort_rates,
    oracle_probability,
    segment_table,
)


def quadratic_instance(seed, dimension, noise_ratio):
    """Original coefficient distributions; Gaussian noise scaled by Var(g(X))."""

    rng = np.random.default_rng(seed)
    objective_matrix = rng.normal(0, 5, (dimension, dimension))
    constraint_matrix = rng.normal(0, 5, (dimension, dimension))
    objective_linear = rng.normal(0, 1, dimension)
    constraint_linear = rng.normal(0, 1, dimension)
    objective_constant, constraint_constant = rng.normal(0, 1, 2)
    symmetric = (constraint_matrix + constraint_matrix.T) / 2
    variance = (
        (4 / 45) * np.sum(np.diag(symmetric) ** 2)
        + (4 / 9) * np.sum(np.triu(symmetric, 1) ** 2)
        + np.sum(constraint_linear**2) / 3
    )
    noise_scale = noise_ratio * np.sqrt(variance)

    def objective(x):
        return np.einsum("ni,ij,nj->n", x, objective_matrix, x) + x @ objective_linear + objective_constant

    def constraint_mean(x):
        return np.einsum("ni,ij,nj->n", x, constraint_matrix, x) + x @ constraint_linear + constraint_constant

    return objective, constraint_mean, float(noise_scale)


def quadratic_grid(dimension):
    if dimension == 1:
        return np.linspace(-1, 1, 801)[:, None]
    if dimension == 2:
        first, second = np.meshgrid(np.linspace(-1, 1, 81), np.linspace(-1, 1, 81), indexing="ij")
        return np.column_stack((first.ravel(), second.ravel()))
    if dimension == 3:
        coordinates = np.meshgrid(*([np.linspace(-1, 1, 31)] * 3), indexing="ij")
        return np.column_stack([coordinate.ravel() for coordinate in coordinates])
    raise ValueError("only dimensions 1, 2, and 3 are used in this study")


def run_quadratic(seed, dimension, config, training_size, noise_ratio):
    objective, constraint_mean, sigma = quadratic_instance(seed, dimension, noise_ratio)
    train_rng, audit_rng = map(np.random.default_rng, np.random.SeedSequence(seed + 100_000).spawn(2))
    x = train_rng.uniform(-1, 1, size=(training_size, dimension))
    response = constraint_mean(x) + sigma * train_rng.normal(size=training_size)
    grid = quadratic_grid(dimension)
    costs = objective(grid)
    exact_risk = norm.cdf(constraint_mean(grid) / sigma)
    feasible = np.flatnonzero(exact_risk <= config.alpha)
    if not len(feasible):
        return {"seed": seed, "dimension": dimension, "status": "mesh_infeasible",
                "noise_scale": sigma, "minimum_mesh_risk": float(exact_risk.min())}

    def simulate(decision, size, rng):
        return float(constraint_mean(decision[None, :])[0]) + sigma * rng.normal(size=size)

    result = run_certified_search(
        candidates=grid, costs=costs, initial_x=x, initial_response=response,
        simulator=simulate, rng=audit_rng, config=config,
    )
    for audit in result["audits"]:
        audit["exact_risk"] = float(norm.cdf(constraint_mean(np.asarray(audit["decision"])[None, :])[0] / sigma))
    result.update({"seed": seed, "dimension": dimension, "status": "mesh_feasible",
                   "noise_scale": sigma, "oracle_mesh_cost": float(costs[feasible].min()),
                   "minimum_mesh_risk": float(exact_risk.min())})
    return result


def tuition_features(sat, gpa, batch_size, merit):
    merit = np.asarray(merit, dtype=float).reshape(-1)
    return np.column_stack((np.full(len(merit), sat), np.full(len(merit), gpa),
                            np.full(len(merit), batch_size), merit))


def tuition_risk(sat, gpa, batch_size, merit, theta):
    probability = oracle_probability(sat, gpa, merit)
    required = int(np.ceil(theta * batch_size))
    return binom.cdf(required - 1, batch_size, probability)


def run_tuition(seed, config, training_size, theta, merit_step):
    training = generate_cohort_rates(training_size, seed=seed)
    x = training[["SAT", "GPA", "batch_size", "merit"]].to_numpy(dtype=float)
    response = theta - training["enrollment_rate"].to_numpy(dtype=float)
    segments = segment_table()
    segment_config = replace(config, delta=config.delta / len(segments))
    merit = np.linspace(0, 25, int(round(25 / merit_step)) + 1)
    audit_seeds = np.random.SeedSequence(seed + 200_000).spawn(len(segments))
    records = []
    for row, audit_seed in zip(segments.itertuples(index=False), audit_seeds):
        grid = tuition_features(row.SAT, row.GPA, row.batch_size, merit)
        exact_risk = tuition_risk(row.SAT, row.GPA, row.batch_size, merit, theta)
        feasible = np.flatnonzero(exact_risk <= config.alpha)
        if not len(feasible):
            records.append({"segment": int(row.segment), "status": "mesh_infeasible"})
            continue

        def simulate(decision, size, rng, *, sat=row.SAT, gpa=row.GPA, batch=row.batch_size):
            probability = float(oracle_probability(sat, gpa, decision[3]))
            return theta - rng.binomial(batch, probability, size=size) / batch

        result = run_certified_search(
            candidates=grid, costs=merit, initial_x=x, initial_response=response,
            simulator=simulate, rng=np.random.default_rng(audit_seed), config=segment_config,
        )
        for audit in result["audits"]:
            audit["exact_risk"] = float(tuition_risk(
                row.SAT, row.GPA, row.batch_size, audit["decision"][3], theta
            ))
        result.update({"segment": int(row.segment), "status": "mesh_feasible",
                       "batch_size": int(row.batch_size),
                       "oracle_mesh_merit_thousands": float(merit[feasible[0]])})
        records.append(result)
    return {"seed": seed, "theta": theta, "segments": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("quadratic1", "quadratic2", "quadratic3", "tuition"), required=True)
    parser.add_argument("--proposal-model", choices=("neural", "tree"), default="neural")
    parser.add_argument("--proposal-quantile", type=float, default=.95)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--training-size", type=int, default=2000)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=600)
    parser.add_argument("--noise-ratio", type=float, default=.2)
    parser.add_argument("--theta", type=float, default=.4)
    parser.add_argument("--merit-step", type=float, default=.1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.seeds < 1 or args.training_size < 2 or args.noise_ratio <= 0:
        parser.error("seeds and training size must be positive; noise ratio must be positive")
    if not 0 < args.theta < 1 or args.merit_step <= 0 or 25 / args.merit_step % 1 > 1e-8:
        parser.error("theta must be a probability and merit step must divide 25")
    config = SearchConfig(
        alpha=.1, delta=.05, rounds=args.rounds, batch_size=args.batch_size,
        proposal_quantile=args.proposal_quantile, proposal_model=args.proposal_model,
        min_separation=.001,
    )
    runs = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        if args.case == "tuition":
            runs.append(run_tuition(seed, config, args.training_size, args.theta, args.merit_step))
        else:
            runs.append(run_quadratic(seed, int(args.case[-1]), config,
                                      args.training_size, args.noise_ratio))
    payload = {"case": args.case, "settings": {**vars(args), "output": str(args.output) if args.output else None},
               "risk_alpha": config.alpha, "confidence_delta": config.delta,
               "runs": runs}
    rendered = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
