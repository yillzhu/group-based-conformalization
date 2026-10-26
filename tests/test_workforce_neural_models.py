import importlib.util
import unittest

import numpy as np

from conformal_cl.workforce.neural_models import (
    fit_relu_mean,
    fit_relu_quantiles,
)


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch is optional")
class WorkforceNeuralModelTests(unittest.TestCase):
    def test_small_relu_models_fit_and_predict(self):
        rng = np.random.default_rng(4)
        x = rng.uniform(-1.0, 1.0, size=(100, 2))
        y = np.maximum(x[:, 0], 0.0) - x[:, 1]
        point = fit_relu_mean(x, y, hidden_units=4, epochs=20, seed=3)
        lower, upper = fit_relu_quantiles(
            x, y, 0.1, 0.9, hidden_units=4, epochs=20, seed=4
        )
        for model in (point, lower, upper):
            prediction = model.predict(x)
            self.assertEqual(prediction.shape, (100,))
            self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue(np.all(lower.predict(x) <= upper.predict(x)))


if __name__ == "__main__":
    unittest.main()
