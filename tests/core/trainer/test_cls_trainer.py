import copy
import os
import shutil
import unittest

from box import Box

from autocare_dlt.core.trainer import ClassificationTrainer as Trainer
from autocare_dlt.core.dataset import build_datasets
from autocare_dlt.core.model import build_model
from autocare_dlt.utils.config import parsing_config



class TestClassificationTrainer(unittest.TestCase):
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

        self.trainer = Trainer(self.model, self.datasets, self.dummy_cfg)
        self.assertTrue(os.path.exists(self.trainer.output_path))
        with self.assertRaises(ValueError):
            cfg = copy.deepcopy(self.dummy_cfg)
            cfg.pop("optim")
            trainer = Trainer(self.model, self.datasets, cfg)

    def test_run_trainer(self):
        self.trainer = Trainer(self.model, self.datasets, self.dummy_cfg)
        ckpt_path = os.path.join(self.trainer.output_path, "best_ckpt.pth")
        self.assertFalse(os.path.exists(ckpt_path))
        self.trainer.train()
        self.assertTrue(os.path.exists(ckpt_path))
