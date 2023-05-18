import unittest

import torch

from autocare_dlt.core.model.head import ClassificationHead as Head


class TestClsHead(unittest.TestCase):
    def setUp(self):
        self.num_classes = 10
        self.batch_size = 2
        self.dummy_cfg = dict(in_channels=512, num_classes=self.num_classes)
        self.dummy_input = torch.rand(self.batch_size, 512)
        self.dummy_labels = torch.LongTensor([[1]] * self.batch_size)

        self.img_size = [512, 512]

    def tearDown(self):
        pass

    def test_build_head(self):
        head = Head(**self.dummy_cfg)
        self.assertEqual(head.num_classes, self.num_classes)
        self.assertTrue(hasattr(head, "head"))

    def test_run_head(self):
        head = Head(**self.dummy_cfg)
        head.train()

        res = head(self.dummy_input)
        self.assertEqual(len(res), 1)

        head.eval()
        res_infer = head(self.dummy_input)
        self.assertEqual(
            list(res_infer[0].size()),
            [self.batch_size, self.num_classes, 1, 1],
        )
