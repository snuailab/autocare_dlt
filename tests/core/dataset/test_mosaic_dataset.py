import unittest

import numpy as np
import torch
from attrdict import AttrDict

from autocare_dlt.core.dataset import MosaicDataset


class TestMosaicDataset(unittest.TestCase):
    def setUp(self):
        self.img_size = [512]
        self.dummy_cfg = AttrDict(
            {
                "img_size": self.img_size,
                "classes": [
                    "person",
                    "bicycle",
                    "car",
                    "motorcycle",
                    "airplane",
                    "bus",
                    "train",
                    "truck",
                    "boat",
                    "traffic light",
                    "fire hydrant",
                    "stop sign",
                    "parking meter",
                    "bench",
                    "bird",
                    "cat",
                    "dog",
                    "horse",
                    "sheep",
                    "cow",
                    "elephant",
                    "bear",
                    "zebra",
                    "giraffe",
                    "backpack",
                    "umbrella",
                    "handbag",
                    "tie",
                    "suitcase",
                    "frisbee",
                    "skis",
                    "snowboard",
                    "sports ball",
                    "kite",
                    "baseball bat",
                    "baseball glove",
                    "skateboard",
                    "surfboard",
                    "tennis racket",
                    "bottle",
                    "wine glass",
                    "cup",
                    "fork",
                    "knife",
                    "spoon",
                    "bowl",
                    "banana",
                    "apple",
                    "sandwich",
                    "orange",
                    "broccoli",
                    "carrot",
                    "hot dog",
                    "pizza",
                    "donut",
                    "cake",
                    "chair",
                    "couch",
                    "potted plant",
                    "bed",
                    "dining table",
                    "toilet",
                    "tv",
                    "laptop",
                    "mouse",
                    "remote",
                    "keyboard",
                    "cell phone",
                    "microwave",
                    "oven",
                    "toaster",
                    "sink",
                    "refrigerator",
                    "book",
                    "clock",
                    "vase",
                    "scissors",
                    "teddy bear",
                    "hair drier",
                    "toothbrush",
                ],
            }
        )
        self.dummy_task_cfg = AttrDict(
            {
                "type": "MosaicDataset",
                "data_root": "tests/assets/detection/small_coco/img",
                "ann": "tests/assets/detection/small_coco/annotation.json",
                "augmentation": {
                    "HorizontalFlip": {"p": 0.5},
                    "ImageNormalization": {"type": "base"},
                },
            }
        )

    def tearDown(self):
        pass

    def test_build_dataset(self):
        dataset = MosaicDataset(self.dummy_cfg, self.dummy_task_cfg)

        self.assertEqual(len(dataset), 16)

    def test_getitem(self):
        dataset = MosaicDataset(self.dummy_cfg, self.dummy_task_cfg)
        idx = np.random.randint(15)
        img, labels = dataset.__getitem__(idx)

        self.assertEqual(len(labels), 3)
        self.assertTrue(torch.is_tensor(img))
        for i, n in enumerate([3, self.img_size[0] * 2, self.img_size[0] * 2]):
            self.assertEqual(img.size()[i], n)
