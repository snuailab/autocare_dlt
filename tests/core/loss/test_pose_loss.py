import codecs
import json
import unittest

import numpy as np
import torch

from autocare_dlt.core.loss import JointsMSELoss


class TestJointsMSELoss(unittest.TestCase):
    def setUp(self):
        super().__init__()

        # === loss cfgs === #
        self.dummy_cfg = {"scaled_loss": False}

        # === load dummy prediction from the json file === #
        fake_preds_text = codecs.open(
            "tests/assets/pose/small_coco_pose/fake_preds.json",
            "r",
            encoding="utf-8",
        ).read()
        fake_preds = json.loads(fake_preds_text)
        self.dummy_preds = torch.from_numpy(
            np.array(fake_preds)
        )  # reverse to the original type

        # === load dummy targets from the json file === #
        fake_targets_text = codecs.open(
            "tests/assets/pose/small_coco_pose/fake_targets.json",
            "r",
            encoding="utf-8",
        ).read()
        fake_targets = json.loads(fake_targets_text)
        for fake_target in fake_targets:  # reverse to the original type
            for key, val in fake_target.items():
                if key == "heatmap" and isinstance(val, list):
                    fake_target[key] = torch.from_numpy(np.array(val))
                if key == "joints" and isinstance(val, list):
                    fake_target[key] = torch.from_numpy(np.array(val))
                if key == "raw_box" and isinstance(val, list):
                    fake_target[key] = np.array(val)
        self.dummy_targets = fake_targets

        # === load dummy wrong targets from the json file === #
        fake_wrong_targets_text = codecs.open(
            "tests/assets/pose/small_coco_pose/fake_wrong_targets.json",
            "r",
            encoding="utf-8",
        ).read()
        fake_wrong_targets = json.loads(fake_wrong_targets_text)
        for (
            fake_wrong_target
        ) in fake_wrong_targets:  # reverse to the original type
            for key, val in fake_wrong_target.items():
                if key == "heatmap" and isinstance(val, list):
                    fake_target[key] = torch.from_numpy(np.array(val))
                if key == "joints" and isinstance(val, list):
                    fake_target[key] = torch.from_numpy(np.array(val))
                if key == "raw_box" and isinstance(val, list):
                    fake_target[key] = np.array(val)
        self.dummy_wrong_targets = fake_wrong_targets

    def tearDown(self):
        pass

    def test_create_loss(self):
        joints_mse_loss = JointsMSELoss(**self.dummy_cfg)
        self.assertIsInstance(joints_mse_loss, JointsMSELoss)
        self.assertEqual(
            joints_mse_loss.scaled_loss, self.dummy_cfg["scaled_loss"]
        )

        with self.assertRaises(KeyError):
            wrong_scaled_loss = {"scaled_loss": "dummy"}
            JointsMSELoss(**wrong_scaled_loss)

    def test_run_loss(self):
        joints_mse_loss = JointsMSELoss(**self.dummy_cfg)
        loss_dict = joints_mse_loss(self.dummy_preds, self.dummy_targets)
        for loss_name, loss_val in loss_dict.items():
            self.assertGreater(1e-4, loss_val.cpu().detach().numpy() - 0.0034)

        with self.assertRaises(ValueError):
            joints_mse_loss(self.dummy_preds, self.dummy_wrong_targets)


if __name__ == "__main__":
    unittest.main()
