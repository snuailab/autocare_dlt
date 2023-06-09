import unittest

import numpy as np
import torch
from box import Box

from autocare_dlt.core.dataset import COCOBaseDataset


class TestCOCOBaseDataset(unittest.TestCase):
    def setUp(self):
        self.img_size = [224]
        self.dummy_cfg = Box(
            {
                "task": "Classifier",
                "img_size": self.img_size,
                "classes": ["dog", "cat"],
            }
        )
        self.fail_cfg = Box(
            {
                "task": "Classifier",
                "img_size": 224,
                "classes": ["dog", "cat"],
            }
        )
        self.dummy_task_cfg = Box(
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
