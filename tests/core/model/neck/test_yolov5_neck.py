import unittest

import torch
from torch import nn

from autocare_dlt.core.model.neck import YOLOv5Neck


class TestYOLOv5Neck(unittest.TestCase):
    def setUp(self):
        self.dummy_cfg = dict(model_size="s")

        self.dummy_input_list = [
            torch.rand(4, 128, 64, 64),
            torch.rand(4, 256, 32, 32),
            torch.rand(4, 256, 16, 16),
        ]
        self.out_channels = [128, 256, 512]
        self.dummy_input_dict = {
            str(n): f for n, f in enumerate(self.dummy_input_list)
        }

    def tearDown(self):
        pass

    def test_build_neck(self):
        with self.assertRaises(ValueError):
            wrong_model_size = dict(model_size="q")
            YOLOv5Neck(**wrong_model_size)

    def test_run_neck(self):
        neck = YOLOv5Neck(**self.dummy_cfg)
        res_input = neck(self.dummy_input_list)

        # Actually we don't need to check below instance.
        # res_list_input is changed by list in yolov5 detector.
        self.assertIsInstance(res_input, tuple)
        self.assertTrue(torch.is_tensor(res_input[0]))
        for i, res in enumerate(res_input):
            self.assertEqual(res.size()[1], self.out_channels[i])
