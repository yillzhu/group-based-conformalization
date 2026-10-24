import unittest

import numpy as np

from conformal_cl.workforce.queueing import (
    QueueConfig,
    QueueScenario,
    sample_scenario,
    service_level_shortfall,
    simulate_service_levels,
)


class WorkforceQueueingTests(unittest.TestCase):
    def test_scenario_generation_is_reproducible(self):
        first = sample_scenario(np.random.default_rng(9), QueueConfig())
        second = sample_scenario(np.random.default_rng(9), QueueConfig())
        np.testing.assert_array_equal(first.arrival_times, second.arrival_times)
        np.testing.assert_array_equal(first.classes, second.classes)
        np.testing.assert_array_equal(first.service_times, second.service_times)

    def test_known_waiting_times(self):
        scenario = QueueScenario(
            arrival_times=np.array([0.0, 0.1, 0.2]),
            classes=np.array([0, 0, 1]),
            service_times=np.array([1.0, 1.0, 1.0]),
        )
        config = QueueConfig(answer_threshold_minutes=0.5)
        levels = simulate_service_levels(scenario, 1, 0, config)
        np.testing.assert_allclose(levels, [0.5, 1.0])

    def test_more_capacity_improves_a_coupled_scenario(self):
        scenario = sample_scenario(np.random.default_rng(12), QueueConfig())
        low = service_level_shortfall(scenario, 2, 0, QueueConfig())
        high = service_level_shortfall(scenario, 3, 1, QueueConfig())
        self.assertLessEqual(high, low)


if __name__ == "__main__":
    unittest.main()
