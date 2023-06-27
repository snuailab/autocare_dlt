import unittest
from box import Box

from autocare_dlt.utils.config import parsing_config


class TestDetectionTrainer(unittest.TestCase):
    def setUp(self):
        self.args = Box(
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
    
    def tearDown(self):
        pass

    def test_parsing_config(self):
        self.dummy_cfg = parsing_config(self.args)
        