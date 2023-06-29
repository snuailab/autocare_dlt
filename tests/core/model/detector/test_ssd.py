import unittest

import torch

from autocare_dlt.core.model.detector import SSD
from autocare_dlt.core.utils import det_labels_to_cuda


class TestSSD(unittest.TestCase):
    def setUp(self):
        self.dummy_model_cfg = {
            "backbone": {"name": "resnet18", "out_indices": [3, 4]},
            "neck": {
                "name": "SSDNeck",
                "out_channels": [256, 512, 512, 256, 256, 128],
                "in_channels": [256, 512],
            },
            "head": {
                "name": "SSDHead",
                "in_channels": [256, 512, 512, 256, 256, 128],
                "aspect_ratio": [
                    [0.5, 2],
                    [0.33, 0.5, 2, 3],
                    [0.33, 0.5, 2, 3],
                    [0.5, 2],
                    [0.5, 2],
                    [0.5, 2],
                ],
                "scales": [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
                "num_classes": 10,
            },
        }

        self.dummy_img = torch.rand(4, 3, 512, 512)
        self.dummy_targets = [
            {"boxes": torch.rand(3, 4), "labels": torch.tensor([3, 6, 1])},
            {"boxes": torch.rand(1, 4), "labels": torch.tensor([0])},
            {"boxes": torch.rand(1, 4), "labels": torch.tensor([2])},
            {"boxes": torch.rand(1, 4), "labels": torch.tensor([0])},
        ]

    def tearDown(self):
        pass

    def test_build_ssd(self):
        from autocare_dlt.core.model.head import SSDHead
        from autocare_dlt.core.model.neck import SSDNeck

        detector = SSD(model_cfg=self.dummy_model_cfg)
        self.assertIsInstance(detector.neck, SSDNeck)
        self.assertIsInstance(detector.head, SSDHead)

    def test_run_ssd(self):
        detector = SSD(model_cfg=self.dummy_model_cfg)
        detector.train()
        pred = detector(self.dummy_img)
        self.assertIsInstance(pred, dict)
        self.assertEqual(len(pred), 3)

    def test_run_ssd_infer(self):
        detector = SSD(model_cfg=self.dummy_model_cfg)
        detector.eval()
        res_infer = detector(self.dummy_img)
        self.assertEqual(len(res_infer), 3)
