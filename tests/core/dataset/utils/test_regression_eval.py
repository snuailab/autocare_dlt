import math
import unittest

import numpy as np
import torch
from box import Box

from autocare_dlt.core.dataset.utils.regression_eval import (
    get_mae,
    get_mse,
    get_rmse,
)


class TestRegressionEval(unittest.TestCase):
    def setUp(self):
        self.dummy_preds = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        self.dummy_labels = torch.tensor([3.0, 3.0, 3.0, 3.0, 3.0])

        self.eps = 1e-6

    def test_get_mae(self):
        self.assertLess(
            get_mae(self.dummy_preds, self.dummy_labels) - 6 / 5, self.eps
        )

    def test_get_mse(self):
        self.assertLess(
            get_mse(self.dummy_preds, self.dummy_labels) - 10 / 5, self.eps
        )

    def test_get_rmse(self):
        self.assertLess(
            get_rmse(self.dummy_preds, self.dummy_labels) - math.sqrt(10 / 5),
            self.eps,
        )
