import unittest

from autocare_dlt.core.utils import LRScheduler


class TestLRScheduler(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_build(self):
        dummy_lr = 0.1
        dummy_iters_per_epoch = 100
        dummy_total_epochs = 10
        with self.assertRaises(ValueError):
            lr_scheduler = LRScheduler(
                "wrong_type",
                dummy_lr,
                dummy_iters_per_epoch,
                dummy_total_epochs,
            )

    def test_warmup(self):
        dummy_lr_cfg = dict(warmup=True, warmup_epochs=2)
        dummy_lr = 0.1
        dummy_iters_per_epoch = 100
        dummy_total_epochs = 10

        lr_scheduler = LRScheduler(
            None,
            dummy_lr,
            dummy_iters_per_epoch,
            dummy_total_epochs,
            **dummy_lr_cfg
        )
        self.assertEqual(lr_scheduler.warmup_total_iters, 200)
        self.assertEqual(lr_scheduler.warmup_lr_start, 1e-6)
        self.assertEqual(lr_scheduler.update_lr(0), 1e-6)
        self.assertEqual(lr_scheduler.update_lr(200), dummy_lr)

        with self.assertRaises(ValueError):
            wrong_lr_cfg = dict(warmup=True)
            lr_scheduler = LRScheduler(
                None,
                dummy_lr,
                dummy_iters_per_epoch,
                dummy_total_epochs,
                **wrong_lr_cfg
            )
        with self.assertRaises(ValueError):
            wrong_lr_cfg = dict(warmup=True, warmup_epochs="a")
            lr_scheduler = LRScheduler(
                None,
                dummy_lr,
                dummy_iters_per_epoch,
                dummy_total_epochs,
                **wrong_lr_cfg
            )

    def test_step_decay(self):
        dummy_lr_cfg = dict(steps=[3, 5], decay=0.1)
        dummy_lr = 0.1
        dummy_iters_per_epoch = 100
        dummy_total_epochs = 10

        lr_scheduler = LRScheduler(
            "step",
            dummy_lr,
            dummy_iters_per_epoch,
            dummy_total_epochs,
            **dummy_lr_cfg
        )
        self.assertEqual(
            lr_scheduler.update_lr(300), dummy_lr * dummy_lr_cfg["decay"]
        )

        with self.assertRaises(ValueError):
            wrong_lr_cfg = dict(steps=[3, 5])
            lr_scheduler = LRScheduler(
                "step",
                dummy_lr,
                dummy_iters_per_epoch,
                dummy_total_epochs,
                **wrong_lr_cfg
            )

        with self.assertRaises(ValueError):
            wrong_lr_cfg = dict(decay=0.1)
            lr_scheduler = LRScheduler(
                "step",
                dummy_lr,
                dummy_iters_per_epoch,
                dummy_total_epochs,
                **wrong_lr_cfg
            )

        with self.assertRaises(ValueError):
            wrong_lr_cfg = dict(steps=3, decay=0.1)
            lr_scheduler = LRScheduler(
                "step",
                dummy_lr,
                dummy_iters_per_epoch,
                dummy_total_epochs,
                **wrong_lr_cfg
            )

        with self.assertRaises(ValueError):
            wrong_lr_cfg = dict(steps=[3, 5], decay=2)
            lr_scheduler = LRScheduler(
                "step",
                dummy_lr,
                dummy_iters_per_epoch,
                dummy_total_epochs,
                **wrong_lr_cfg
            )

    def test_cosine_decay(self):
        dummy_lr = 0.1
        dummy_iters_per_epoch = 100
        dummy_total_epochs = 10
        lr_scheduler = LRScheduler(
            "cosine", dummy_lr, dummy_iters_per_epoch, dummy_total_epochs
        )
        self.assertEqual(lr_scheduler.update_lr(500), dummy_lr * 0.5)

    def test_linear_decay(self):
        dummy_lr = 0.1
        dummy_iters_per_epoch = 100
        dummy_total_epochs = 10
        lr_scheduler = LRScheduler(
            "linear", dummy_lr, dummy_iters_per_epoch, dummy_total_epochs
        )
        self.assertEqual(lr_scheduler.update_lr(500), dummy_lr * 0.5)
