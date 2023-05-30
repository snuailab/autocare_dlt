import unittest

import numpy as np
import torch
from box import Box

from autocare_dlt.core.dataset import COCOPoseDataset


class TestCOCOPoseDataset(unittest.TestCase):
    def setUp(self):
        self.img_size = [512]
        self.dummy_cfg = Box(
            {
                "img_size": self.img_size,
                "classes": [
                    "nose",
                    "left_eye",
                    "right_eye",
                    "left_ear",
                    "right_ear",
                    "left_shoulder",
                    "right_shoulder",
                    "left_elbow",
                    "right_elbow",
                    "left_wrist",
                    "right_wrist",
                    "left_hip",
                    "right_hip",
                    "left_knee",
                    "right_knee",
                    "left_ankle",
                    "right_ankle",
                ],
            }
        )
        self.fail_cfg = Box(
            {
                "img_size": 512,
            }
        )
        self.dummy_task_cfg = Box(
            {
                "type": "PoseCOCODetectionDataset",
                "data_root": "tests/assets/pose/small_coco_pose/img",
                "ann": "tests/assets/pose/small_coco_pose/annotation.json",
                "augmentation": {
                    "HorizontalFlip": {"p": 0.5},
                    "ImageNormalization": {"type": "base"},
                },
            }
        )

    def tearDown(self):
        pass

    def test_validate_img_size(self):
        self.assertRaises(
            ValueError,
            lambda: COCOPoseDataset(
                self.fail_cfg,
                self.dummy_task_cfg,
            ),
        )

    def test_build_coco_pose_dataset(self):
        coco_pose_dataset = COCOPoseDataset(
            self.dummy_cfg,
            self.dummy_task_cfg,
        )

        self.assertEqual(
            len(coco_pose_dataset),
            16,
            msg="num of imgs for unittest not equal to 16",
        )

    def test_getitem(self):
        coco_pose_dataset = COCOPoseDataset(
            self.dummy_cfg,
            self.dummy_task_cfg,
        )
        idx = np.random.randint(15)
        img, labels = coco_pose_dataset.__getitem__(idx)

        self.assertEqual(len(labels), 7, "label meta not equal to 7")
        self.assertTrue(torch.is_tensor(img))
        for i, n in enumerate([3, self.img_size[0], self.img_size[0]]):
            self.assertEqual(img.size()[i], n)
