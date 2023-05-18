import unittest

import torch

from autocare_dlt.core.model.head import SSDHead4Point
from autocare_dlt.core.utils.boxes import bboxes_iou


class TestSSDHead4Point(unittest.TestCase):
    def setUp(self):
        self.num_classes = 1
        self.dummy_cfg = dict(
            in_channels=[512, 512, 1024, 512, 256, 256],
            num_classes=self.num_classes,
            aspect_ratio=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
            scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
            topk_candidates=400,
        )
        self.dummy_input = [
            torch.rand(2, 512, 64, 64),
            torch.rand(2, 512, 32, 32),
            torch.rand(2, 1024, 16, 16),
            torch.rand(2, 512, 8, 8),
            torch.rand(2, 256, 4, 4),
            torch.rand(2, 256, 2, 2),
        ]
        self.dummy_labels = [
            {"boxes": torch.rand(3, 8), "labels": torch.tensor([0, 0, 0])},
            {"boxes": torch.rand(1, 8), "labels": torch.tensor([0])},
        ]
        self.img_size = [512, 512]

    def tearDown(self):
        pass

    def test_build_head(self):
        head = SSDHead4Point(**self.dummy_cfg)
        self.assertEqual(head.num_classes, self.num_classes + 1)  # add BG cls

    def test_run_head(self):
        head = SSDHead4Point(**self.dummy_cfg)
        head.train()
        pred = head.forward(self.dummy_input, self.img_size)
        self.assertIsInstance(pred, dict)
        self.assertEqual(len(pred), 3)

        head.eval()
        res_infer = head.forward(self.dummy_input, self.img_size)
        self.assertEqual(len(res_infer), 3)

        num_cand = self.dummy_cfg["topk_candidates"] * self.num_classes
        for r in res_infer:
            self.assertEqual(r.size()[1], num_cand)
        self.assertEqual(len(res_infer[0].size()), 3)
        self.assertEqual(len(res_infer[1].size()), 2)
        self.assertEqual(len(res_infer[2].size()), 2)
