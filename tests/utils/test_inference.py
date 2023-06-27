import sys
import unittest

import cv2
import torch
from box import Box

from autocare_dlt.core.model.classifier import *
from autocare_dlt.core.model.detector import *
from autocare_dlt.core.utils.inference import Inferece
from autocare_dlt.core.dataset import build_datasets
from autocare_dlt.core.model import build_model
from autocare_dlt.utils.config import parsing_config


class TestInferece(unittest.TestCase):
    def setUp(self):
        args = Box(
            {
                "output_dir": "tests/outputs",
                "exp_name": "test_base_trainer",
                "model_cfg":  "tests/assets/detection/configs/retinanet_resnet18.json",
                "data_cfg": "tests/assets/detection/configs/coco_small.json",
                "gpus": "0",
                "num_gpus": 1,
                "world_size": 1,
                "ckpt": None,
                "resume": False,
                "ema": False,
                "overwrite": True,
                "input_size": [512],
            }
        )

        self.dummy_cfg = parsing_config(args)

        self.model, _ = build_model(self.dummy_cfg)
        self.datasets = build_datasets(self.dummy_cfg.data)
        if torch.cuda.is_available():
            self.model.cuda()

    def tearDown(self):
        pass

    def test_inference_single_img(self):
        dummy_input = cv2.imread(
            "tests/assets/detection/small_coco/img/000000118113.jpg"
        )

        infer = Inferece(self.dummy_cfg, single_img=True)
        out = infer(self.model, dummy_input)
        self.assertTrue(isinstance(out, list))
