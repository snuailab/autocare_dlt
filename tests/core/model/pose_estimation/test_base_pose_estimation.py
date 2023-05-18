import copy
import unittest

import torch
from attrdict import AttrDict

from autocare_dlt.core.model.pose_estimation.base_pose_estimation import (
    BasePoseEstimation as Model,
)


class TestBasePoseEstimation(unittest.TestCase):
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

    def test_create_base_pose_estimation(self):

        dummy_model_cfg = copy.deepcopy(
            self.dummy_model_cfg
        )  # because the ``Model()`` changes the ``self.dummy_model_cfg``
        model = Model(dummy_model_cfg)
        self.assertIsInstance(model.backbone, torch.nn.Module)

        from autocare_dlt.core.model.neck import Identity

        self.assertIsInstance(model.neck, Identity)

        from autocare_dlt.core.model.head import PoseHead

        self.assertIsInstance(model.head, PoseHead)
