import unittest

import torch
from torch import nn

from autocare_dlt.core.loss import BCE_FocalLoss, CE_FocalLoss


class TestBCE_FocalLoss(unittest.TestCase):
    def setUp(self):
        self.dummy_cfg = dict(alpha=0.25, gamma=1.5, reduction="mean")
        self.dummy_preds = torch.tensor(
            [[0.1, 0.1, 0.1, 0.7], [0.7, 0.1, 0.1, 0.1]]
        )
        self.dummy_targets = torch.tensor([3, 0])
        self.dummy_targets_onehot = torch.tensor(
            [[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]]
        )

    def tearDown(self):
        pass

    def test_create_loss(self):
        focal_loss = BCE_FocalLoss(**self.dummy_cfg)
        self.assertIsInstance(focal_loss, BCE_FocalLoss)
        self.assertIsInstance(focal_loss.bce, nn.BCEWithLogitsLoss)
        self.assertEqual(focal_loss.gamma, self.dummy_cfg["gamma"])
        self.assertEqual(focal_loss.alpha, self.dummy_cfg["alpha"])
        self.assertEqual(focal_loss.reduction, self.dummy_cfg["reduction"])

        with self.assertRaises(ValueError):
            wrong_gamma = {"gamma": -1.0}
            BCE_FocalLoss(**wrong_gamma)
        with self.assertRaises(ValueError):
            wrong_alpha = {"alpha": 2.0}
            BCE_FocalLoss(**wrong_alpha)
        with self.assertRaises(KeyError):
            wrong_reduction = {"reduction": None}
            BCE_FocalLoss(**wrong_reduction)

    def test_run_loss(self):
        focal_loss = BCE_FocalLoss(**self.dummy_cfg)
        res = focal_loss(self.dummy_preds, self.dummy_targets)
        res_onehot = focal_loss(self.dummy_preds, self.dummy_targets_onehot)
        self.assertEqual(res, res_onehot)


class TestCE_FocalLoss(unittest.TestCase):
    def setUp(self):
        self.dummy_cfg = dict(gamma=0, size_average=True, ignore_index=-1)
        self.dummy_preds = torch.tensor(
            [[0.1, 0.1, 0.1, 0.7], [0.7, 0.1, 0.1, 0.1]]
        )
        self.dummy_targets = torch.tensor([3, 0])

    def tearDown(self):
        pass

    def test_create_loss(self):
        focal_loss = CE_FocalLoss(**self.dummy_cfg)
        self.assertIsInstance(focal_loss, CE_FocalLoss)
        self.assertEqual(focal_loss.gamma, self.dummy_cfg["gamma"])
        self.assertEqual(
            focal_loss.size_average, self.dummy_cfg["size_average"]
        )
        self.assertEqual(
            focal_loss.ignore_index, self.dummy_cfg["ignore_index"]
        )

        with self.assertRaises(ValueError):
            wrong_gamma = {"gamma": -1.0}
            CE_FocalLoss(**wrong_gamma)

    def test_run_loss(self):

        focal_loss = CE_FocalLoss(**self.dummy_cfg)
        res = focal_loss(self.dummy_preds, self.dummy_targets)

        ignore_preds = torch.tensor(
            [[0.1, 0.1, 0.1, 0.7], [0.7, 0.1, 0.1, 0.1], [0.7, 0.1, 0.1, 0.1]]
        )
        ignore_targets = torch.tensor([3, 0, -1])
        res_ignore = focal_loss(ignore_preds, ignore_targets)
        self.assertEqual(res, res_ignore)


if __name__ == "__main__":
    unittest.main()
