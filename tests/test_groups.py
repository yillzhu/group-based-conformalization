import unittest

import numpy as np

from conformal_cl.groups import Group


class GroupTests(unittest.TestCase):
    def test_default_bounds_are_not_shared(self):
        first = Group()
        second = Group()
        first.bounds[(1.0, 0.0)] = 1
        self.assertEqual(second.bounds, {})

    def test_split_creates_independent_regions_and_preserves_parent(self):
        parent = Group(bounds={(1.0, 10.0): 1}, qhat=0.25)
        parent.min_x = np.array([-1.0])
        parent.max_x = np.array([1.0])
        parent.ul_x = np.array([0.0])
        parent.min_v = -1.0
        parent.max_v = 1.0
        parent.ul_v = 0.0

        cal_x = np.linspace(-1, 1, 100).reshape(-1, 1)
        scores = np.linspace(0, 1, 100)
        cut = np.array([1.0, 0.0])
        left, right = parent.split(cut, cal_x, scores)

        self.assertIsNot(parent.bounds, left.bounds)
        self.assertIsNot(parent.bounds, right.bounds)
        self.assertIsNot(left.bounds, right.bounds)
        self.assertNotIn(tuple(cut), parent.bounds)
        self.assertEqual(left.bounds[tuple(cut)], -1)
        self.assertEqual(right.bounds[tuple(cut)], 1)

        # Region-dependent extrema must be recomputed, not shifted from parent.
        self.assertTrue(np.isnan(left.min_x))
        self.assertTrue(np.isnan(left.max_x))
        self.assertTrue(np.isnan(right.min_x))
        self.assertTrue(np.isnan(right.max_x))

    def test_insufficient_child_inherits_parent_correction(self):
        parent = Group(qhat=0.75)
        cal_x = np.array([[-1.0], [1.0]])
        scores = np.array([0.1, 0.2])
        left, right = parent.split(np.array([1.0, 0.0]), cal_x, scores)
        self.assertFalse(left.Xhat)
        self.assertFalse(right.Xhat)
        self.assertEqual(left.qhat, 0.75)
        self.assertEqual(right.qhat, 0.75)


if __name__ == "__main__":
    unittest.main()
