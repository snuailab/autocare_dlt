import unittest

import torch

from autocare_dlt.core.model.head import PoseHead


class TestPoseHead(unittest.TestCase):
    def setUp(self):
        super().__init__()

        self.num_classes = 16
        self.dummy_cfg = dict(
            in_channels=256,
            num_classes=self.num_classes,
        )
        self.dummy_input = torch.rand(16, 256, 1, 1)
        self.img_size = [512, 512]

    def tearDown(self):
        pass

    def test_build_head(self):
        head = PoseHead(**self.dummy_cfg)
        with self.assertRaises(ValueError):
            wrong_in_channels = dict(in_channels=256.0, num_classes=10)
            PoseHead(**wrong_in_channels)

    def test_run_head(self):
        head = PoseHead(**self.dummy_cfg)
        head.train()
        pred = head(self.dummy_input)
        self.assertIsInstance(pred, torch.Tensor)
        self.assertEqual(len(pred), 16)

        head.eval()
        res_infer = head(self.dummy_input)
        self.assertEqual(res_infer.shape[0], 16)
        self.assertEqual(res_infer.shape[1], 16)
