import unittest

import torch

from autocare_dlt.core.model.head import SSDHead


class TestSSDHead(unittest.TestCase):
    def setUp(self):
        self.num_classes = 10
        self.dummy_cfg = dict(
            in_channels=[256, 512, 512, 256, 256, 128],
            num_classes=self.num_classes,
            aspect_ratio=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
            scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
            topk_candidates=400,
        )
        self.dummy_input = [
            torch.rand(2, 256, 64, 64),
            torch.rand(2, 512, 32, 32),
            torch.rand(2, 512, 16, 16),
            torch.rand(2, 256, 8, 8),
            torch.rand(2, 256, 8, 8),
            torch.rand(2, 128, 8, 8),
        ]
        self.dummy_labels = [
            {"boxes": torch.rand(3, 4), "labels": torch.tensor([3, 6, 1])},
            {"boxes": torch.rand(1, 4), "labels": torch.tensor([0])},
        ]
        self.img_size = [512, 512]

    def tearDown(self):
        pass

    def test_build_head(self):
        head = SSDHead(**self.dummy_cfg)
        self.assertEqual(head.num_classes, self.num_classes + 1)  # add BG cls

    def test_run_head(self):
        head = SSDHead(**self.dummy_cfg)
        head.train()
        pred = head(self.dummy_input, self.img_size)
        self.assertIsInstance(pred, dict)
        self.assertEqual(len(pred), 3)

        head.eval()
        res_infer = head(self.dummy_input, self.img_size)
        self.assertEqual(len(res_infer), 3)

        num_cand = self.dummy_cfg["topk_candidates"] * self.num_classes
        for r in res_infer:
            self.assertEqual(r.size()[1], num_cand)
        self.assertEqual(len(res_infer[0].size()), 3)
        self.assertEqual(len(res_infer[1].size()), 2)
        self.assertEqual(len(res_infer[2].size()), 2)
