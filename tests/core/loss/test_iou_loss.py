import unittest

import torch

from autocare_dlt.core.loss import IOUloss


class TestIOUloss(unittest.TestCase):
    def setUp(self):
        self.dummy_cfg = {"reduction": "mean", "loss_type": "giou"}
        self.dummy_preds = torch.tensor(
            [[0.1, 0.1, 0.3, 0.3], [0.2, 0.2, 0.4, 0.4]]
        )
        self.dummy_targets = torch.tensor(
            [[0.2, 0.2, 0.4, 0.4], [0.3, 0.3, 0.5, 0.5]]
        )

    def tearDown(self):
        pass

    def test_create_loss(self):
        iou_loss = IOUloss(**self.dummy_cfg)
        self.assertIsInstance(iou_loss, IOUloss)
        self.assertEqual(iou_loss.reduction, self.dummy_cfg["reduction"])
        self.assertEqual(iou_loss.loss_type, self.dummy_cfg["loss_type"])

        with self.assertRaises(KeyError):
            wrong_loss_type = {"reduction": "mean", "loss_type": "dummy"}
            IOUloss(**wrong_loss_type)
        with self.assertRaises(KeyError):
            wrong_redunction = {"reduction": "dummy", "loss_type": "giou"}
            IOUloss(**wrong_redunction)

    def test_run_loss(self):
        iou_loss = IOUloss(**self.dummy_cfg)
        res = iou_loss(self.dummy_preds, self.dummy_targets)
        self.assertGreater(1e-4, res.item() - 1.7460)

        with self.assertRaises(ValueError):
            wrong_target = torch.tensor([[0.2, 0.2, 0.4, 0.4]])
            iou_loss(self.dummy_preds, wrong_target)


if __name__ == "__main__":
    unittest.main()
