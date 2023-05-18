import unittest

import torch

from autocare_dlt.core.model.head import YOLOv5Head


class TestYOLOv5Head(unittest.TestCase):
    def setUp(self):
        self.num_classes = 80
        self.dummy_cfg = dict(
            model_size="s",
            num_classes=self.num_classes,
            anchors=[
                [10, 13, 16, 30, 33, 23],
                [30, 61, 62, 45, 59, 119],
                [116, 90, 156, 198, 373, 326],
            ],
            topk=1000,
        )
        # for small model
        self.dummy_input = [
            torch.rand(4, 128, 64, 64),
            torch.rand(4, 256, 32, 32),
            torch.rand(4, 512, 16, 16),
        ]
        self.dummy_labels = [
            {"boxes": torch.rand(3, 4), "labels": torch.tensor([3, 6, 1])},
            {"boxes": torch.rand(1, 4), "labels": torch.tensor([0])},
        ]
        self.img_size = [512, 512]

    def tearDown(self):
        pass

    def test_build_head(self):
        with self.assertRaises(ValueError):
            wrong_model_size = dict(model_size="q", num_classes=10)
            YOLOv5Head(**wrong_model_size)

    def test_run_head(self):
        head = YOLOv5Head(**self.dummy_cfg)
        head.train()
        pred = head(self.dummy_input.copy(), img_size=self.img_size)
        self.assertIsInstance(pred, list)

        head.eval()
        res_infer = head(self.dummy_input.copy(), img_size=self.img_size)
        self.assertEqual(len(res_infer), 3)

        num_cand = self.dummy_cfg["topk"]
        for r in res_infer:
            self.assertEqual(r.size()[1], num_cand)
        self.assertEqual(len(res_infer[0].size()), 3)
        self.assertEqual(len(res_infer[1].size()), 2)
        self.assertEqual(len(res_infer[2].size()), 2)
