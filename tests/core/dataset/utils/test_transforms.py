import unittest

import numpy as np
import torch

from autocare_dlt.core.dataset.utils.transforms import *


class TestTransforms(unittest.TestCase):
    def setUp(self):
        self.num_classes = 10
        self.dummy_cfg = dict(
            in_channels=[256, 512, 512, 256, 256, 128],
            num_classes=self.num_classes,
            aspect_ratio=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
            scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
            iou_thres=0.45,
            topk_candidates=400,
            neg_to_pos_ratio=3,
        )
        self.dummy_input = np.random.randint(0, 255, (512, 512, 3))
        self.dummy_labels = [
            {"boxes": torch.rand(3, 4), "labels": torch.tensor([3, 6, 1])},
            {"boxes": torch.rand(1, 4), "labels": torch.tensor([0])},
        ]
        self.img_size = [512, 512]

    def tearDown(self):
        pass

    def test_build_head(self):
        pass
        # head = Transforms(**self.dummy_cfg)
        # self.assertEqual(head.num_classes, self.num_classes+1) # add BG cls

    def test_run_head(self):
        pass
