import unittest

import torch

from autocare_dlt.core.model.detector import RetinaNet
from autocare_dlt.core.utils import det_labels_to_cuda


class TestRetinaNet(unittest.TestCase):
    def setUp(self):
        self.dummy_model_cfg = {
            "backbone": {"name": "resnet18", "out_indices": [2, 3, 4]},
            "neck": {
                "name": "FeaturePyramidNetwork",
                "in_channels": [128, 256, 512],
                "out_channels": 256,
            },
            "head": {
                "name": "RetinaNetHead",
                "in_channels": 256,
                "aspect_ratio": [0.5, 1.0, 2.0],
                "anchor_size": [32, 64, 128, 256, 512],
                "topk_candidates": 1000,
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

    def test_build_RetinaNet(self):
        from autocare_dlt.core.model.head import RetinaNetHead
        from autocare_dlt.core.model.neck import FeaturePyramidNetwork

        detector = RetinaNet(model_cfg=self.dummy_model_cfg)
        self.assertIsInstance(detector.neck, FeaturePyramidNetwork)
        self.assertIsInstance(detector.head, RetinaNetHead)

    def test_run_RetinaNet(self):
        detector = RetinaNet(model_cfg=self.dummy_model_cfg)
        detector.train()
        pred = detector(self.dummy_img)
        self.assertIsInstance(pred, dict)
        self.assertEqual(len(pred), 3)

    def test_run_RetinaNet_infer(self):
        detector = RetinaNet(model_cfg=self.dummy_model_cfg)
        detector.eval()
        res_infer = detector(self.dummy_img)
        self.assertEqual(len(res_infer), 3)
