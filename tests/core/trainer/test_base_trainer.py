import copy
import os
import shutil
import unittest

from box import Box

from autocare_dlt.core.trainer import BaseTrainer
from autocare_dlt.core.dataset import build_datasets
from autocare_dlt.core.model import build_model
from autocare_dlt.utils.config import parsing_config


class TestBaseTrainer(unittest.TestCase):
    def setUp(self):
        args = Box(
            {
                "output_dir": "tests/outputs",
                "exp_name": "test_base_trainer",
                "model_cfg": "tests/assets/classification/configs/classifier.json",
                "data_cfg": "tests/assets/classification/configs/cat_and_dog_coco.json",
                "gpus": "0",
                "num_gpus": 1,
                "world_size": 1,
                "ckpt": None,
                "resume": False,
                "ema": False,
                "overwrite": True,
            }
        )

        self.dummy_cfg = parsing_config(args)

        self.model, _ = build_model(self.dummy_cfg)
        self.datasets = build_datasets(self.dummy_cfg.data)

    def tearDown(self):
        shutil.rmtree(self.trainer.output_path)
        del self.trainer

    def test_build_trainer(self):
        self.trainer = BaseTrainer(self.model, self.datasets, self.dummy_cfg)
        self.assertTrue(os.path.exists(self.trainer.output_path))
        with self.assertRaises(ValueError):
            cfg = copy.deepcopy(self.dummy_cfg)
            cfg.pop("optim")
            trainer = BaseTrainer(self.model, self.datasets, cfg)
        with self.assertRaises(ValueError):
            cfg = copy.deepcopy(self.dummy_cfg)
            cfg.update({"task": "wrong_task"})
            trainer = BaseTrainer(self.model, self.datasets, cfg)

    def test_check_loss_fn(self):
        self.trainer = BaseTrainer(self.model, self.datasets, self.dummy_cfg)
        self.assertTrue(hasattr(self.dummy_cfg, "loss"))
        self.assertTrue(hasattr(self.trainer, "_get_loss_fn"))

        self.trainer._get_loss_fn()
        self.assertTrue(hasattr(self.trainer, "loss_manager"))
