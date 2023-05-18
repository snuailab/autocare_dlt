import unittest

from autocare_dlt.core.model.detector import BaseDetector


class TestBaseDetector(unittest.TestCase):
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

    def tearDown(self):
        pass

    def test_build_detector(self):
        from autocare_dlt.core.model.head import SSDHead
        from autocare_dlt.core.model.neck import SSDNeck

        detector = BaseDetector(self.dummy_model_cfg)
        self.assertIsInstance(detector.neck, SSDNeck)
        self.assertIsInstance(detector.head, SSDHead)
