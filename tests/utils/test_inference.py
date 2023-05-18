import sys
import unittest

import cv2
import torch

from autocare_dlt.core.model import build_model
from autocare_dlt.core.model.classifier import *
from autocare_dlt.core.model.detector import *
from autocare_dlt.core.utils.inference import Inferece
from autocare_dlt.utils.config import (
    classifier_list,
    detector_list,
    json_to_dict,
    regressor_list,
)


class TestInferece(unittest.TestCase):
    def setUp(self):
        self.model_cfg = (
            "tests/assets/detection/configs/retinanet_resnet18.json"
        )
        self.data_cfg = "tests/assets/detection/configs/coco_small.json"
        self.dummy_cfg = {}
        self.dummy_cfg.update(json_to_dict(self.model_cfg))
        self.dummy_cfg.update({"input_size": [512]})

    def tearDown(self):
        pass

    def test_inference_single_img(self):
        dummy_input = cv2.imread(
            "tests/assets/detection/small_coco/img/000000118113.jpg"
        )
        model, _ = build_model(self.dummy_cfg)
        infer = Inferece(self.dummy_cfg, single_img=True)
        out = infer(model, dummy_input)
        self.assertTrue(isinstance(out, list))
