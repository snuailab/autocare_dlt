import unittest

import torch

from autocare_dlt.core.model.head import RetinaNetHead


class TestRetinaNetHead(unittest.TestCase):
    def setUp(self):
        self.num_classes = 10
        self.dummy_cfg = dict(
            in_channels=256,
            num_classes=self.num_classes,
            aspect_ratio=[0.5, 1.0, 2.0],
            anchor_size=[32, 64, 128, 256, 512],
            topk_candidates=1000,
        )
        self.dummy_input = [
            torch.rand(2, 256, 64, 64),
            torch.rand(2, 256, 32, 32),
            torch.rand(2, 256, 16, 16),
            torch.rand(2, 256, 8, 8),
            torch.rand(2, 256, 4, 4),
        ]
        self.dummy_labels = [
            {"boxes": torch.rand(3, 4), "labels": torch.tensor([3, 6, 1])},
            {"boxes": torch.rand(1, 4), "labels": torch.tensor([0])},
        ]
        self.img_size = [512, 512]

    def tearDown(self):
        pass

    def test_build_head(self):
        head = RetinaNetHead(**self.dummy_cfg)
        with self.assertRaises(ValueError):
            wrong_in_channels = dict(in_channels=256.0, num_classes=10)
            RetinaNetHead(**wrong_in_channels)

    def test_run_head(self):
        head = RetinaNetHead(**self.dummy_cfg)
        head.train()
        pred = head(self.dummy_input, self.img_size)
        self.assertIsInstance(pred, dict)
        self.assertEqual(len(pred), 3)

        head.eval()
        res_infer = head(self.dummy_input, self.img_size)
        self.assertEqual(len(res_infer), 3)

        num_cand = self.dummy_cfg["topk_candidates"] * len(
            self.dummy_cfg["anchor_size"]
        )
        for r in res_infer:
            self.assertEqual(r.size()[1], num_cand)
        self.assertEqual(len(res_infer[0].size()), 3)
        self.assertEqual(len(res_infer[1].size()), 2)
        self.assertEqual(len(res_infer[2].size()), 2)
