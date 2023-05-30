import unittest

import torch
from box import Box

from autocare_dlt.core.model.classifier.base_classifier import \
    BaseClassifier as Model

class TestBaseClassifier(unittest.TestCase):
    def setUp(self):

        self.model_cfg = Box(
            {
                "backbone": {"name": "resnet18"},
                "neck": {"name": "Identity"},
                "head": {
                    "name": "ClassificationHead",
                    "in_channels": 512,
                    "num_classes": 10,
                },
            }
        )

    def test_create_base_classifier(self):

        with self.assertRaises(TypeError):
            Model(self.model_cfg)

        model = Model(self.model_cfg)

        self.assertIsInstance(model.backbone, torch.nn.Module)

        from autocare_dlt.core.model.neck import Identity

        self.assertIsInstance(model.neck, Identity)

        from autocare_dlt.core.model.head import ClassificationHead

        self.assertIsInstance(model.head, ClassificationHead)
