"""Known-oracle, cohort-level tuition experiment for localized conformal bounds.

This is a research prototype, not a replacement for the historical JANOS
reproduction.  It trains intervals directly on observed cohort enrollment rates.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import binom
from sklearn.preprocessing import StandardScaler

from conformal_cl.enrollment.models import (
    create_enrollment_net,
    interval_score_loss,
)
from conformal_cl.enrollment.oracle_tuition import (
    generate_cohort_rates,
    oracle_probability,
    project_merit_monotone_,
    segment_table,
)
from conformal_cl.order_stats import conformal_order_statistic


def create_monotone_interval_net(hidden_size=20):
    return nn.Sequential(nn.Linear(4, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 2))


def prepare_features(frame, scaler=None):
    features = frame[
        ["SAT", "GPA", "batch_size", "merit"]
    ].to_numpy(dtype=float, copy=True)
    if scaler is None:
        scaler = StandardScaler().fit(features[:, :3])
    features[:, :3] = scaler.transform(features[:, :3])
    return features.astype(np.float32), scaler


def train_models(train, *, seed, alpha, point_epochs=300, interval_epochs=1200):
    features, scaler = prepare_features(train)
    target = train["enrollment_rate"].to_numpy(dtype=np.float32)[:, None]
    weights = train["batch_size"].to_numpy(dtype=np.float32)[:, None]
    x = torch.from_numpy(features)
    y = torch.from_numpy(target)
    w = torch.from_numpy(weights)

    torch.manual_seed(seed)
    point = create_enrollment_net(input_size=4, hidden_size=20)
    optimizer = torch.optim.Adam(point.parameters(), lr=0.01)
    for _ in range(point_epochs):
        prediction = point(x)
        losses = nn.functional.binary_cross_entropy(prediction, y, reduction="none")
        loss = torch.sum(losses * w) / torch.sum(w)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        project_merit_monotone_(point, merit_index=3)

    torch.manual_seed(seed + 1)
    interval = create_monotone_interval_net()
    optimizer = torch.optim.Adam(interval.parameters(), lr=0.005)
    for _ in range(interval_epochs):
        prediction = interval(x)
        crossing = torch.relu(prediction[:, 0] - prediction[:, 1]).mean()
        loss = interval_score_loss(prediction, y, alpha) + 20.0 * crossing
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        project_merit_monotone_(interval, merit_index=3)
    point.eval()
    interval.eval()
    return point, interval, scaler


def predict(model, features):
    with torch.no_grad():
        return model(torch.from_numpy(features.astype(np.float32))).numpy()


def calibration_scores(interval, features, target):
    bounds = predict(interval, features)
    y = np.asarray(target, dtype=float).reshape(-1, 1)
    return np.maximum(y - bounds[:, 1:2], bounds[:, 0:1] - y).reshape(-1)


def first_feasible(grid, values, threshold):
    feasible = np.flatnonzero(values >= threshold)
    return float(grid[feasible[0]]) if feasible.size else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--theta", type=float, default=0.4)
    parser.add_argument("--train-rows", type=int, default=12000)
    parser.add_argument("--calibration-rows", type=int, default=12000)
    parser.add_argument("--grid-step", type=float, default=0.01)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    train = generate_cohort_rates(args.train_rows, seed=args.seed)
    calibration = generate_cohort_rates(args.calibration_rows, seed=args.seed + 10_000)
    point, interval, scaler = train_models(
        train, seed=args.seed, alpha=args.alpha
    )
    cal_x, _ = prepare_features(calibration, scaler)
    scores = calibration_scores(interval, cal_x, calibration["enrollment_rate"])
    qhat_global = conformal_order_statistic(scores, alpha=args.alpha)

    grid = np.arange(0.0, 25.0 + args.grid_step / 2, args.grid_step)
    segments = segment_table()
    decisions = []
    for row in segments.itertuples(index=False):
        raw = np.column_stack(
            [
                np.full_like(grid, row.SAT),
                np.full_like(grid, row.GPA),
                np.full_like(grid, row.batch_size),
                grid,
            ]
        )
        raw[:, :3] = scaler.transform(raw[:, :3])
        point_grid = predict(point, raw).reshape(-1)
        interval_grid = predict(interval, raw)
        lower_grid = interval_grid[:, 0]

        crossing_index = np.flatnonzero(lower_grid >= args.theta)
        crossing = grid[crossing_index[0]] if crossing_index.size else 12.5
        inner = np.clip([crossing - 3.0, crossing + 3.0], 0.0, 25.0)
        edges = np.unique(np.r_[0.0, inner, 25.0])

        segment_mask = calibration["segment"].to_numpy() == row.segment
        local_adjusted = np.full_like(grid, -np.inf, dtype=float)
        group_records = []
        for group_index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            if group_index == len(edges) - 2:
                cal_mask = segment_mask & calibration["merit"].between(
                    left, right, inclusive="both"
                ).to_numpy()
                grid_mask = (grid >= left) & (grid <= right)
            else:
                cal_mask = segment_mask & calibration["merit"].between(
                    left, right, inclusive="left"
                ).to_numpy()
                grid_mask = (grid >= left) & (grid < right)
            local_qhat = conformal_order_statistic(scores[cal_mask], alpha=args.alpha)
            local_adjusted[grid_mask] = lower_grid[grid_mask] - local_qhat
            group_records.append(
                {
                    "left": float(left),
                    "right": float(right),
                    "n_calibration": int(np.sum(cal_mask)),
                    "qhat": local_qhat,
                }
            )

        probability = oracle_probability(row.SAT, row.GPA, grid)
        required_count = int(np.ceil(args.theta * row.batch_size))
        predictive_success = binom.sf(required_count - 1, row.batch_size, probability)
        choices = {
            "point": first_feasible(grid, point_grid, args.theta),
            "global": first_feasible(
                grid, lower_grid - qhat_global, args.theta
            ),
            "localized": first_feasible(grid, local_adjusted, args.theta),
            "oracle_mean": first_feasible(grid, probability, args.theta),
            "oracle_predictive": first_feasible(
                grid, predictive_success, 1.0 - args.alpha
            ),
        }
        record = {
            "segment": int(row.segment),
            "batch_size": int(row.batch_size),
            "groups": group_records,
        }
        for name, merit in choices.items():
            record[name] = merit
            if merit is not None:
                index = int(round(merit / args.grid_step))
                record[f"{name}_oracle_probability"] = float(probability[index])
                record[f"{name}_predictive_success"] = float(
                    predictive_success[index]
                )
        decisions.append(record)

    method_names = ["point", "global", "localized", "oracle_mean", "oracle_predictive"]
    summary = {}
    segment_weights = np.asarray([record["batch_size"] for record in decisions])
    for name in method_names:
        merits = [record[name] for record in decisions if record[name] is not None]
        probabilities = [
            record[f"{name}_oracle_probability"]
            for record in decisions
            if record[name] is not None
        ]
        successes = [
            record[f"{name}_predictive_success"]
            for record in decisions
            if record[name] is not None
        ]
        summary[name] = {
            "feasible_segments": len(merits),
            "mean_merit_dollars": float(1000 * np.mean(merits)) if merits else None,
            "cohort_weighted_mean_merit_dollars": (
                float(
                    1000
                    * np.average(
                        [record[name] for record in decisions],
                        weights=segment_weights,
                    )
                )
                if len(merits) == len(decisions)
                else None
            ),
            "oracle_mean_violations": int(np.sum(np.asarray(probabilities) < args.theta)),
            "predictive_90pct_violations": int(
                np.sum(np.asarray(successes) < 1.0 - args.alpha)
            ),
        }

    report = {
        "parameters": vars(args),
        "global_qhat": qhat_global,
        "summary": summary,
        "segments": decisions,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w") as stream:
            stream.write(rendered + "\n")


if __name__ == "__main__":
    main()
