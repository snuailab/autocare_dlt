import unittest

import torch

from autocare_dlt.core.utils import AverageMeter


class TestAverageMeter(unittest.TestCase):
    """Compute average for torch.Tensor, used for loss average."""

    def setUp(self):
        self.n_count = 0
        self.sum = 0

    def test_build(self):
        avg = AverageMeter()
        self.assertIsInstance(avg, AverageMeter)

    def test_update(self):
        avg = AverageMeter()
        size = 1
        if torch.cuda.is_available():
            pseudo_tensor = torch.cuda.FloatTensor(size).fill_(2)
        else:
            pseudo_tensor = torch.FloatTensor(size).fill_(2)
        out = avg.update(pseudo_tensor)
        self.assertEqual(avg.sum, 2)
        self.assertEqual(avg.count, 1)

    def test_reset(self):
        avg = AverageMeter()
        avg.reset()
        self.assertEqual(avg.sum, 0)
        self.assertEqual(avg.count, 0)

    def test_avgl(self):
        avg = AverageMeter()
        size = 2
        if torch.cuda.is_available():
            pseudo_tensor = torch.cuda.FloatTensor(size).fill_(2)
        else:
            pseudo_tensor = torch.FloatTensor(size).fill_(2)
        avg.update(pseudo_tensor)
        avg.update(pseudo_tensor)
        result = avg.avg

        self.assertEqual(result, 5)
