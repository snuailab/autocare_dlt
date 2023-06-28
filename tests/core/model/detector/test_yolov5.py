import unittest

import torch

from autocare_dlt.core.model.detector import YOLOv5


class TestYOLOv5(unittest.TestCase):
    def setUp(self):
        self.dummy_model_cfg = {
            "model_size": "s",
            "backbone": {
                "name": "YOLOv5Backbone",
                "focus": False,
                "with_C3TR": False,
            },
            "neck": {"name": "YOLOv5Neck"},
            "head": {
                "name": "YOLOv5Head",
                "anchors": [
                    [10, 13, 16, 30, 33, 23],
                    [30, 61, 62, 45, 59, 119],
                    [116, 90, 156, 198, 373, 326],
                ],
                "num_classes": 80,
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

    def test_build_YOLOv5(self):
        from autocare_dlt.core.model.head import YOLOv5Head
        from autocare_dlt.core.model.neck import YOLOv5Neck

        detector = YOLOv5(model_cfg=self.dummy_model_cfg)
        self.assertIsInstance(detector.neck, YOLOv5Neck)
        self.assertIsInstance(detector.head, YOLOv5Head)

    def test_run_YOLOv5(self):
        detector = YOLOv5(model_cfg=self.dummy_model_cfg)
        detector.train()
        pred = detector(self.dummy_img)
        self.assertIsInstance(pred, list)

    def test_run_YOLOv5_infer(self):
        detector = YOLOv5(model_cfg=self.dummy_model_cfg)
        detector.eval()
        res_infer = detector(self.dummy_img)
        self.assertEqual(len(res_infer), 3)
