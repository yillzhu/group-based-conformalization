"""Compact PGLib-OPF IEEE 39-bus data needed by the DC experiment.

The source case is ``pglib_opf_case39_epri.m`` from PGLib-OPF v23.07.
The IEEE data are licensed under CC BY 4.0.  See
https://github.com/power-grid-lib/pglib-opf and the attribution in that case.
Only active loads, generator bounds/costs, and DC branch fields are retained.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Case39:
    base_mva: float
    loads: np.ndarray
    generator_bus: np.ndarray
    p_min: np.ndarray
    p_max: np.ndarray
    linear_cost: np.ndarray
    branch_from: np.ndarray
    branch_to: np.ndarray
    branch_x: np.ndarray
    branch_rate: np.ndarray
    branch_tap: np.ndarray

    @property
    def demand(self) -> float:
        return float(self.loads.sum())


def load_case39() -> Case39:
    """Return the fields of the PGLib IEEE 39-bus case used here.

    Bus indices are converted from MATPOWER's one-based convention to zero-based
    indices.  A zero transformer ratio in the source means a unit tap ratio.
    """

    loads = np.array([
        97.6, 0.0, 322.0, 500.0, 0.0, 0.0, 233.8, 522.0, 6.5, 0.0,
        0.0, 8.53, 0.0, 0.0, 320.0, 329.0, 0.0, 158.0, 0.0, 680.0,
        274.0, 0.0, 247.5, 308.6, 224.0, 139.0, 281.0, 206.0, 283.5,
        0.0, 9.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1104.0,
    ])
    generator_bus = np.arange(29, 39, dtype=int)
    p_max = np.array([1040., 646., 725., 652., 508., 687., 580., 564., 865., 1100.])
    linear_cost = np.array([
        6.724778, 14.707625, 24.804734, 34.844643, 24.652994,
        32.306483, 18.157477, 31.550181, 22.503168, 27.434444,
    ])
    branch_from = np.array([
        1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 7, 8, 9, 10,
        10, 10, 12, 12, 13, 14, 15, 16, 16, 16, 16, 17, 17, 19, 19,
        20, 21, 22, 22, 23, 23, 25, 25, 26, 26, 26, 28, 29,
    ], dtype=int) - 1
    branch_to = np.array([
        2, 39, 3, 25, 30, 4, 18, 5, 14, 6, 8, 7, 11, 31, 8, 9, 39,
        11, 13, 32, 11, 13, 14, 15, 16, 17, 19, 21, 24, 18, 27, 20,
        33, 34, 22, 23, 35, 24, 36, 26, 37, 27, 28, 29, 29, 38,
    ], dtype=int) - 1
    branch_x = np.array([
        .0411, .025, .0151, .0086, .0181, .0213, .0133, .0128, .0129,
        .0026, .0112, .0092, .0082, .025, .0046, .0363, .025, .0043,
        .0043, .02, .0435, .0435, .0101, .0217, .0094, .0089, .0195,
        .0135, .0059, .0082, .0173, .0138, .0142, .018, .014, .0096,
        .0143, .035, .0272, .0323, .0232, .0147, .0474, .0625, .0151,
        .0156,
    ])
    branch_rate = np.array([
        600., 1000., 500., 500., 900., 500., 500., 600., 500., 1200.,
        900., 900., 480., 1800., 900., 900., 900., 600., 600., 900.,
        500., 500., 600., 600., 600., 600., 600., 600., 600., 600.,
        600., 900., 900., 900., 900., 600., 900., 600., 900., 600.,
        900., 600., 600., 600., 600., 1200.,
    ])
    branch_tap = np.array([
        0., 0., 0., 0., 1.025, 0., 0., 0., 0., 0., 0., 0., 0., 1.07,
        0., 0., 0., 0., 0., 1.07, 1.006, 1.006, 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 1.06, 1.07, 1.009, 0., 0., 1.025, 0., 0., 0.,
        1.025, 0., 0., 0., 0., 1.025,
    ])
    branch_tap[branch_tap == 0.] = 1.
    return Case39(
        100., loads, generator_bus, np.zeros(10), p_max, linear_cost,
        branch_from, branch_to, branch_x, branch_rate, branch_tap,
    )
