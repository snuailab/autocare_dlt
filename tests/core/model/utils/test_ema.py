import unittest

import torch
from torch import nn

from autocare_dlt.core.model.utils.ema import ModelEMA


class TestModelEMA(unittest.TestCase):
    def setUp(self):
        self.dummy_ema = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.dummy_model = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1, bias=False
        )
        nn.init.zeros_(self.dummy_ema.weight)
        nn.init.ones_(self.dummy_model.weight)

    def tearDown(self):
        pass

    def test_build_ema(self):
        decay = 0.99
        max_iter = 0
        ema = ModelEMA(self.dummy_ema, decay, max_iter)
        for p in ema.ema.parameters():
            self.assertFalse(p.requires_grad)

    def test_run_ema(self):
        decay = 0.99
        max_iter = 0
        ema = ModelEMA(self.dummy_ema, decay, max_iter)
        ema.update(self.dummy_model, None, 1)
        for p in ema.ema.parameters():
            self.assertEqual(0.01, p.data[0][0][0][0])
