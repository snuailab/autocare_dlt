import codecs
import json
import unittest

import numpy as np
import torch

from autocare_dlt.core.dataset import *
from autocare_dlt.core.dataset.utils.pose_eval import pck_accuracy


class TestPoseEval(unittest.TestCase):
    def setUp(self):
        super().__init__()

        # === fake cfgs === #
        self.dummy_batch_size = 1
        self.dummy_num_joints = 17
        self.dummy_num_joints_detected = (
            13  # fake prediction has 13 joints detected
        )
        self.dummy_score_max = 1.0
        self.dummy_score_min = 0.0
        self.dummy_dim_joints_loc = 2

        # === load dummy prediction the json file === #
        fake_preds_text = codecs.open(
            "tests/assets/pose/small_coco_pose/fake_preds.json",
            "r",
            encoding="utf-8",
        ).read()
        fake_preds = json.loads(fake_preds_text)
        self.dummy_preds = np.array(fake_preds)

        # === load dummy labels from the json file === #
        fake_labels_text = codecs.open(
            "tests/assets/pose/small_coco_pose/fake_labels.json",
            "r",
            encoding="utf-8",
        ).read()
        fake_labels = json.loads(fake_labels_text)
        tensor_list = [
            "heatmap",
            "joints",
        ]
        ndarr_list = [
            "raw_box",
        ]
        for fake_label in fake_labels:
            for key, val in fake_label.items():
                if key in tensor_list:
                    fake_label[key] = torch.from_numpy(np.array(val))
                elif key in ndarr_list:
                    fake_label[key] = np.array(val)
        self.dummy_labels = fake_labels

    def test_pck_accuracy(self):
        _, score, cnt, pred_joints_loc = pck_accuracy(
            self.dummy_preds,
            self.dummy_labels,
        )

        self.assertLessEqual(score, self.dummy_score_max)
        self.assertGreaterEqual(score, self.dummy_score_min)
        self.assertEqual(cnt, self.dummy_num_joints_detected)
        self.assertEqual(pred_joints_loc.shape[0], self.dummy_batch_size)
        self.assertEqual(pred_joints_loc.shape[1], self.dummy_num_joints)
        self.assertEqual(pred_joints_loc.shape[2], self.dummy_dim_joints_loc)

    # TODO: tailor for the pose
    # def test_convert_keypoints_to_coco(self):
    # res = convert_to_coco_format(self.dummy_preds, self.dataset)
    # self.assertIsInstance(res, list)
    # self.assertEqual(len(res), 17)
    # self.assertEqual(len(res[0]), 4)
