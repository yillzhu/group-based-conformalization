"""Discrete-event simulator for a two-class, multiskill service system."""

from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np


@dataclass(frozen=True)
class QueueScenario:
    arrival_times: np.ndarray
    classes: np.ndarray
    service_times: np.ndarray


@dataclass(frozen=True)
class QueueConfig:
    horizon_minutes: float = 240.0
    arrival_rate_per_hour: float = 36.0
    mean_service_minutes: float = 10.0
    service_cv: float = 0.6
    service_level_target: float = 0.8
    answer_threshold_minutes: float = 1.0
    surge_probability: float = 0.08
    surge_multiplier: float = 1.35
    daily_volume_sigma: float = 0.18
    class_mix_logit_sigma: float = 0.85
    flexible_slowdown: float = 1.05


def sample_scenario(rng: np.random.Generator, config: QueueConfig) -> QueueScenario:
    """Sample one day, including volume, mix, arrivals, and service requirements."""

    volume_multiplier = rng.lognormal(
        -0.5 * config.daily_volume_sigma**2, config.daily_volume_sigma
    )
    if rng.random() < config.surge_probability:
        volume_multiplier *= config.surge_multiplier
    expected_arrivals = (
        config.arrival_rate_per_hour * config.horizon_minutes / 60.0
    ) * volume_multiplier
    count = int(rng.poisson(expected_arrivals))
    arrivals = np.sort(rng.uniform(0.0, config.horizon_minutes, count))
    mix = 1.0 / (
        1.0 + np.exp(-rng.normal(0.0, config.class_mix_logit_sigma))
    )
    classes = (rng.random(count) >= mix).astype(np.int8)

    sigma = np.sqrt(np.log1p(config.service_cv**2))
    log_mean = np.log(config.mean_service_minutes) - 0.5 * sigma**2
    service = rng.lognormal(log_mean, sigma, count)
    return QueueScenario(arrivals, classes, service)


def simulate_service_levels(
    scenario: QueueScenario,
    dedicated_per_class: int,
    flexible: int,
    config: QueueConfig,
) -> np.ndarray:
    """Return the fraction of each class answered within the waiting-time target."""

    if dedicated_per_class < 0 or flexible < 0:
        raise ValueError("staff counts must be nonnegative")
    arrivals = np.asarray(scenario.arrival_times, dtype=float)
    classes = np.asarray(scenario.classes, dtype=np.int8)
    service = np.asarray(scenario.service_times, dtype=float)
    if not (len(arrivals) == len(classes) == len(service)):
        raise ValueError("scenario arrays must have equal lengths")
    if np.any(np.diff(arrivals) < 0.0) or np.any((classes < 0) | (classes > 1)):
        raise ValueError("scenario arrivals must be sorted and classes must be 0 or 1")

    waiting = [[], []]
    wait_head = [0, 0]
    available_dedicated = [int(dedicated_per_class), int(dedicated_per_class)]
    available_flexible = int(flexible)
    completions: list[tuple[float, int, int]] = []
    answered_on_time = np.zeros(2, dtype=int)
    totals = np.bincount(classes, minlength=2)
    arrival_index = 0

    def dispatch(now: float) -> None:
        nonlocal available_flexible
        for request_class in (0, 1):
            while (
                available_dedicated[request_class] > 0
                and wait_head[request_class] < len(waiting[request_class])
            ):
                index = waiting[request_class][wait_head[request_class]]
                wait_head[request_class] += 1
                available_dedicated[request_class] -= 1
                answered_on_time[request_class] += (
                    now - arrivals[index] <= config.answer_threshold_minutes
                )
                heapq.heappush(
                    completions,
                    (now + service[index], 0, request_class),
                )

        while available_flexible > 0:
            heads = [
                waiting[j][wait_head[j]]
                if wait_head[j] < len(waiting[j])
                else None
                for j in (0, 1)
            ]
            if heads[0] is None and heads[1] is None:
                break
            if heads[1] is None or (
                heads[0] is not None and arrivals[heads[0]] <= arrivals[heads[1]]
            ):
                request_class = 0
            else:
                request_class = 1
            index = heads[request_class]
            wait_head[request_class] += 1
            available_flexible -= 1
            answered_on_time[request_class] += (
                now - arrivals[index] <= config.answer_threshold_minutes
            )
            heapq.heappush(
                completions,
                (now + config.flexible_slowdown * service[index], 1, request_class),
            )

    while arrival_index < len(arrivals) or completions:
        next_arrival = (
            arrivals[arrival_index] if arrival_index < len(arrivals) else np.inf
        )
        next_completion = completions[0][0] if completions else np.inf
        now = min(next_arrival, next_completion)

        while completions and completions[0][0] <= now:
            _, pool, request_class = heapq.heappop(completions)
            if pool == 0:
                available_dedicated[request_class] += 1
            else:
                available_flexible += 1
        while arrival_index < len(arrivals) and arrivals[arrival_index] <= now:
            request_class = int(classes[arrival_index])
            waiting[request_class].append(arrival_index)
            arrival_index += 1
        dispatch(now)

    # A zero-arrival class has no observed service failure that day.
    return np.divide(
        answered_on_time,
        totals,
        out=np.ones(2, dtype=float),
        where=totals > 0,
    )


def service_level_shortfall(
    scenario: QueueScenario,
    dedicated_per_class: int,
    flexible: int,
    config: QueueConfig,
) -> float:
    """Positive values violate the daily service-level requirement."""

    levels = simulate_service_levels(
        scenario, dedicated_per_class, flexible, config
    )
    return float(config.service_level_target - np.min(levels))


def sample_queue_responses(
    decisions: np.ndarray,
    rng: np.random.Generator,
    config: QueueConfig,
) -> np.ndarray:
    decisions = np.asarray(decisions, dtype=int)
    if decisions.ndim != 2 or decisions.shape[1] != 2:
        raise ValueError("decisions must have shape (n, 2)")
    result = np.empty(len(decisions))
    for index, (dedicated, flexible) in enumerate(decisions):
        result[index] = service_level_shortfall(
            sample_scenario(rng, config), dedicated, flexible, config
        )
    return result


def queue_violation_probabilities(
    decisions: np.ndarray,
    simulations: int,
    rng: np.random.Generator,
    config: QueueConfig,
) -> np.ndarray:
    """Evaluate decisions on common independently sampled operating days."""

    decisions = np.asarray(decisions, dtype=int)
    scenarios = [sample_scenario(rng, config) for _ in range(simulations)]
    result = np.zeros(len(decisions))
    for scenario in scenarios:
        for index, (dedicated, flexible) in enumerate(decisions):
            result[index] += (
                service_level_shortfall(
                    scenario, dedicated, flexible, config
                )
                > 0.0
            )
    return result / simulations
