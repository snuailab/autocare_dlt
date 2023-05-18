import unittest

import numpy as np
import torch
from attrdict import AttrDict

from autocare_dlt.core.dataset import COCOBaseDataset


class TestCOCOBaseDataset(unittest.TestCase):
    def setUp(self):
        self.img_size = [224]
        self.dummy_cfg = AttrDict(
            {
                "task": "Classifier",
                "img_size": self.img_size,
                "classes": {"animal": ["dog", "cat"]},
            }
        )
        self.fail_cfg = AttrDict(
            {
                "task": "Classifier",
                "img_size": 224,
                "classes": {"animal": ["dog", "cat"]},
            }
        )
        self.dummy_task_cfg = AttrDict(
            {
                "type": "COCOClassificationDataset",
                "data_root": "tests/assets/classification/cat_and_dog/images",
                "ann": "tests/assets/classification/cat_and_dog/coco/coco.json",
                "augmentation": {
                    "HorizontalFlip": {"p": 0.5},
                    "ImageNormalization": {"type": "base"},
                },
            }
        )

    def tearDown(self):
        pass

    def test_validate_img_size(self):
        self.assertRaises(
            ValueError,
            lambda: COCOBaseDataset(self.fail_cfg, self.dummy_task_cfg),
        )
