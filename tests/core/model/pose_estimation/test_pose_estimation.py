import copy
import unittest

import torch
from attrdict import AttrDict

from autocare_dlt.core.model.pose_estimation.pose_estimation import (
    PoseEstimation as Model,
)


class TestPoseEstimation(unittest.TestCase):
    def setUp(self):
        super().__init__()

        self.dummy_model_cfg = AttrDict(
            {
                "backbone": {
                    "name": "hrnet_w32",
                    "out_indices": [1],
                    "feature_location": "",
                },
                "neck": {"name": "Identity"},
                "head": {
                    "name": "PoseHead",
                    "in_channels": 32,
                    "sigma": 3,
                    "num_classes": 17,
                },
            }
        )
        self.dummy_input = torch.randn(1, 3, 64, 64)

    def test_create_pose_estimator(self):

        dummy_model_cfg = copy.deepcopy(
            self.dummy_model_cfg
        )  # because the ``Model()`` changes the ``self.dummy_model_cfg``
        model = Model(dummy_model_cfg)
        self.assertIsInstance(model.backbone, torch.nn.Module)

        from autocare_dlt.core.model.neck import Identity

        self.assertIsInstance(model.neck, Identity)

        from autocare_dlt.core.model.head import PoseHead

        self.assertIsInstance(model.head, PoseHead)

    def test_training_forward(self):

        dummy_model_cfg = copy.deepcopy(
            self.dummy_model_cfg
        )  # because the ``Model()`` changes the ``self.dummy_model_cfg``
        model = Model(dummy_model_cfg)
        model.train()
        outputs = model.forward(self.dummy_input)

        self.assertIsInstance(outputs, torch.Tensor)
        self.assertEqual(
            outputs.shape[0], 1
        )  # because the dummy has 1 batch size
        self.assertEqual(
            outputs.shape[1], 17
        )  # because the model predicts 17 keypoints

    def test_eval_forward(self):

        dummy_model_cfg = copy.deepcopy(
            self.dummy_model_cfg
        )  # because the ``Model()`` changes the ``self.dummy_model_cfg``
        model = Model(dummy_model_cfg)
        model.eval()
        outputs = model.forward(self.dummy_input)

        self.assertIsInstance(outputs, torch.Tensor)
        self.assertEqual(
            outputs.shape[0], 1
        )  # because the dummy has 1 batch size
        self.assertEqual(
            outputs.shape[1], 17
        )  # because the model predicts 17 keypoints
