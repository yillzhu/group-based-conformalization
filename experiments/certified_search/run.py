#!/usr/bin/env python3
"""Non-RCC, boosted-quantile proposal search with fresh pointwise audits."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from conformal_cl.certified_search import SearchConfig, run_certified_search
from conformal_cl.transparent.capacity import CASES
from conformal_cl.workforce.continuous_queue import (
    ContinuousQueueConfig,
    approximate_continuous_queue_oracle,
    continuous_queue_violation_probabilities,
    sample_continuous_queue_dataset,
    sample_continuous_queue_responses,
)


def picture_mean(x):
    return .18 + .55 * x + .23 * np.sin(2 * np.pi * x) + .16 * x**2


def picture_scale(x):
    return np.asarray((.1, .2, .4, .8))[np.searchsorted((.25, .5, .75), x, side="right")]


def picture_risk(x):
    return np.clip((.35 - picture_mean(x) + picture_scale(x)) / (2 * picture_scale(x)), 0, 1)


def run_picture(seed, config, training_size):
    train_rng, audit_rng = map(np.random.default_rng, np.random.SeedSequence(seed).spawn(2))
    x = train_rng.uniform(0, 1, training_size)
    response = .35 - picture_mean(x) - picture_scale(x) * train_rng.uniform(-1, 1, training_size)
    grid = np.linspace(0, 1, 1001)[:, None]

    def simulate(decision, size, rng):
        return .35 - picture_mean(decision[0]) - picture_scale(decision[0]) * rng.uniform(-1, 1, size)

    result = run_certified_search(
        candidates=grid, costs=grid[:, 0], initial_x=x[:, None],
        initial_response=response, simulator=simulate, rng=audit_rng, config=config,
    )
    add_risk(result, lambda decision: float(picture_risk(decision[0])))
    oracle_grid = grid[np.flatnonzero(picture_risk(grid[:, 0]) <= config.alpha)[0], 0]
    result["oracle_cost_on_mesh"] = float(oracle_grid)
    return result


def run_capacity(seed, config, training_size):
    case = next(case for case in CASES if case.name == "interior_quiet_region")
    train_rng, audit_rng = map(np.random.default_rng, np.random.SeedSequence(seed).spawn(2))
    x, residual = case.draw(training_size, train_rng)
    grid = np.linspace(case.edges[0], case.edges[-1], 2001)[:, None]

    def simulate(decision, size, rng):
        return 1 - decision[0] + case.scale(decision[0]) * rng.uniform(-1, 1, size)

    result = run_certified_search(
        candidates=grid, costs=grid[:, 0], initial_x=x[:, None],
        initial_response=1 - x + residual, simulator=simulate,
        rng=audit_rng, config=config,
    )
    add_risk(result, lambda decision: float(case.risk(decision[0])))
    result["oracle_cost_on_mesh"] = float(grid[np.flatnonzero(case.risk(grid[:, 0]) <= config.alpha)[0], 0])
    return result


def run_queue(seed, config, training_size, evaluation_size, *, structured=False):
    queue = ContinuousQueueConfig()
    lower, upper = np.array((4., 0.)), np.array((12., 9.))
    train_rng, audit_rng, evaluation_rng, oracle_rng = map(
        np.random.default_rng, np.random.SeedSequence(seed).spawn(4)
    )
    train_x, response = sample_continuous_queue_dataset(
        lower, upper, training_size, train_rng, queue
    )
    dedicated, flexible = np.meshgrid(
        np.linspace(lower[0], upper[0], 81),
        np.linspace(lower[1], upper[1], 91), indexing="ij",
    )
    grid = np.column_stack((dedicated.ravel(), flexible.ravel()))
    costs = queue.dedicated_cost * grid[:, 0] + queue.flexible_cost * grid[:, 1]

    def simulate(decision, size, rng):
        return sample_continuous_queue_responses(np.repeat(decision[None, :], size, axis=0), rng, queue)

    result = run_certified_search(
        candidates=grid, costs=costs, initial_x=train_x, initial_response=response,
        simulator=simulate, rng=audit_rng,
        config=replace(config, capacity_index=1) if structured else config,
    )
    if result["audits"]:
        risks = continuous_queue_violation_probabilities(
            np.asarray([audit["decision"] for audit in result["audits"]]),
            evaluation_size, evaluation_rng, queue,
        )
        for audit, risk in zip(result["audits"], risks):
            audit["evaluation_risk"] = float(risk)
    oracle_decision, oracle_cost, oracle_risk = approximate_continuous_queue_oracle(
        domain_lower=lower, domain_upper=upper, alpha=config.alpha,
        simulations=evaluation_size, rng=oracle_rng, config=queue,
        dedicated_grid_size=401,
    )
    result["approximate_oracle"] = {
        "decision": oracle_decision.tolist(), "cost": oracle_cost,
        "same_sample_risk": oracle_risk,
    }
    return result


def add_risk(result, risk):
    for audit in result["audits"]:
        audit["evaluation_risk"] = risk(audit["decision"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("picture", "capacity", "queue", "all"), default="all")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--training-size", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=600)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--proposal-quantile", type=float, default=.95)
    parser.add_argument("--queue-structured", action="store_true",
                        help="learn the random requirement by dedicated FTE, using the exact -flexible FTE response shift")
    parser.add_argument("--evaluation-size", type=int, default=30000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = SearchConfig(rounds=args.rounds, batch_size=args.batch_size,
                          proposal_quantile=args.proposal_quantile,
                          min_separation=.01)
    cases = ("picture", "capacity", "queue") if args.case == "all" else (args.case,)
    runs = {}
    for case in cases:
        function = {"picture": run_picture, "capacity": run_capacity, "queue": run_queue}[case]
        records = []
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            record = (function(seed, config, args.training_size, args.evaluation_size,
                               structured=args.queue_structured)
                      if case == "queue" else function(seed, config, args.training_size))
            records.append(record)
        runs[case] = records
    payload = {"method": "boosted_quantile_proposals_with_independent_binomial_audits",
               "config": vars(args) | {"output": str(args.output) if args.output else None},
               "risk_alpha": config.alpha, "confidence_delta": config.delta,
               "results": runs}
    result_json = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result_json + "\n")
    print(result_json)


if __name__ == "__main__":
    main()
