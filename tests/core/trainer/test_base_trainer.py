import copy
import os
import shutil
import unittest

from box import Box

from autocare_dlt.core.trainer import BaseTrainer


class TestBaseTrainer(unittest.TestCase):
    def setUp(self):
        self.dummy_cfg = Box(
            {
                "task": "Classifier",
                "model": {"head": {}},
                "loss": {
                    "cls_loss": {
                        "name": "CE_FocalLoss",
                        "params": {
                            "gamma": 2,
                            "size_average": True,
                            "ignore_index": -1,
                        },
                    }
                },
                "data": {},
                "optim": {"name": "SGD"},
                "num_classes": 3,
                "classes": ["1", "2", "3"],
                "exp_name": "test_base_trainer",
                "ema": False,
                "num_gpus": 1,
            }
        )

    def tearDown(self):
        shutil.rmtree(self.trainer.log_path)
        shutil.rmtree(self.trainer.output_path)
        del self.trainer

    def test_build_trainer(self):
        self.trainer = BaseTrainer(self.dummy_cfg)
        self.assertTrue(os.path.exists(self.trainer.log_path))
        self.assertTrue(os.path.exists(self.trainer.output_path))
        self.assertEqual(
            self.trainer.cfg["model"]["head"]["num_classes"],
            self.dummy_cfg.num_classes,
        )
        with self.assertRaises(ValueError):
            cfg = copy.deepcopy(self.dummy_cfg)
            cfg.pop("optim")
            trainer = BaseTrainer(cfg)
        with self.assertRaises(ValueError):
            cfg = copy.deepcopy(self.dummy_cfg)
            cfg.update({"task": "wrong_task"})
            trainer = BaseTrainer(cfg)

    def test_check_loss_fn(self):
        self.trainer = BaseTrainer(self.dummy_cfg)
        self.assertTrue(hasattr(self.dummy_cfg, "loss"))
        self.assertTrue(hasattr(self.trainer, "_get_loss_fn"))

        self.trainer._get_loss_fn()
        self.assertTrue(hasattr(self.trainer, "loss_manager"))
