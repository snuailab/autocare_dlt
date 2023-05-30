import unittest

import torch
from box import Box

from autocare_dlt.core.model.classifier.classifier import Classifier as Model


class TestClassifier(unittest.TestCase):
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

        self.dummy_input = torch.randn(2, 3, 64, 64)
        self.dummy_target = torch.LongTensor([[1], [1]])

    def test_create_classifier(self):

        with self.assertRaises(TypeError):
            Model(model_cfg=self.model_cfg)

        model = Model(model_cfg=self.model_cfg)

        self.assertIsInstance(model.backbone, torch.nn.Module)

        from autocare_dlt.core.model.neck import Identity

        self.assertIsInstance(model.neck, Identity)

        from autocare_dlt.core.model.head import ClassificationHead

        self.assertIsInstance(model.head, ClassificationHead)

    def test_training_forward(self):

        model = Model(model_cfg=self.model_cfg)

        # train
        model.train()

        outputs = model.forward(self.dummy_input)

        self.assertIsInstance(outputs, list)

    def test_eval_forward(self):

        model = Model(model_cfg=self.model_cfg)

        # eval
        model.eval()

        outputs = model.forward(self.dummy_input)

        self.assertIsInstance(outputs, list)
        self.assertEqual(list(outputs[0].size()), [2, 10, 1, 1])
