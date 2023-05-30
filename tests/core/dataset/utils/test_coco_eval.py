import json
import math
import sys
import unittest
from collections import OrderedDict

import numpy as np
import torch
from box import Box

from autocare_dlt.core.dataset import *
from autocare_dlt.core.dataset.utils import (
    coco_evaluation,
    convert_to_coco_format,
)
from autocare_dlt.core.utils import nms


class TestCocoEval(unittest.TestCase):
    def setUp(self):
        self.img_size = [512]
        self.dummy_data_cfg = Box(
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
        self.dummy_task_cfg = Box(
            {
                "type": "COCODetectionDataset",
                "data_root": "tests/assets/detection/small_coco/img",
                "ann": "tests/assets/detection/small_coco/annotation.json",
                "augmentation": {
                    "HorizontalFlip": {"p": 0.5},
                    "ImageNormalization": {"type": "base"},
                },
            }
        )
        with open(
            "./tests/assets/detection/small_coco/SSD_res18_base_inf_result.json"
        ) as json_file:
            self.dummy_preds_coco = json.load(
                json_file, object_pairs_hook=OrderedDict
            )
        self.dataset = getattr(
            sys.modules[__name__], self.dummy_task_cfg.type
        )(self.dummy_data_cfg, self.dummy_task_cfg)
        self.dummy_preds = [
            {
                "boxes": torch.tensor([[0.5434, 0.2523, 0.7639, 0.7101]]),
                "scores": torch.tensor([0.9896]),
                "labels": torch.tensor([0]),
            },
            {
                "boxes": torch.tensor(
                    [
                        [0.5795, 0.3360, 0.9880, 0.8807],
                        [0.3802, 0.7503, 0.6791, 0.8499],
                    ]
                ),
                "scores": torch.tensor([0.9610, 0.3281]),
                "labels": torch.tensor([0, 43]),
            },
            {
                "boxes": torch.tensor(
                    [[2.5787e-01, 3.2486e-01, 5.9471e-01, 7.8843e-01]]
                ),
                "scores": torch.tensor([0.9981]),
                "labels": torch.tensor([0]),
            },
            {
                "boxes": torch.tensor([[0.5187, 0.7525, 0.8490, 0.9353]]),
                "scores": torch.tensor([0.9354]),
                "labels": torch.tensor([66]),
            },
            {
                "boxes": torch.tensor(
                    [[1.9219e-01, 2.4047e-01, 7.1588e-01, 9.7711e-01]]
                ),
                "scores": torch.tensor([0.9872]),
                "labels": torch.tensor([0]),
            },
            {
                "boxes": torch.tensor([[0.6809, 0.2188, 0.8977, 0.8379]]),
                "scores": torch.tensor([0.9821]),
                "labels": torch.tensor([72]),
            },
            {
                "boxes": torch.tensor([[0.0055, 0.1611, 0.5952, 0.7966]]),
                "scores": torch.tensor([0.8929]),
                "labels": torch.tensor([0]),
            },
            {
                "boxes": torch.tensor([[0.3045, 0.5533, 0.4444, 0.6427]]),
                "scores": torch.tensor([0.5214]),
                "labels": torch.tensor([45]),
            },
            {
                "boxes": torch.tensor([[0.0216, 0.2478, 0.4436, 0.9219]]),
                "scores": torch.tensor([0.9635]),
                "labels": torch.tensor([0]),
            },
            {
                "boxes": torch.tensor([[0.0972, 0.2937, 0.4919, 0.7791]]),
                "scores": torch.tensor([0.9990]),
                "labels": torch.tensor([0]),
            },
            {
                "boxes": torch.tensor([[0.5352, 0.3373, 0.6268, 0.3783]]),
                "scores": torch.tensor([0.1972]),
                "labels": torch.tensor([73]),
            },
            {
                "boxes": torch.tensor([[0.1826, 0.3376, 0.3629, 0.4462]]),
                "scores": torch.tensor([0.8671]),
                "labels": torch.tensor([68]),
            },
            {
                "boxes": torch.tensor(
                    [[2.6635e-01, 6.4096e-01, 4.3022e-01, 7.6787e-01]]
                ),
                "scores": torch.tensor([0.7603]),
                "labels": torch.tensor([61]),
            },
            {
                "boxes": torch.tensor([[0.4873, 0.5500, 0.6051, 0.8983]]),
                "scores": torch.tensor([0.8823]),
                "labels": torch.tensor([0]),
            },
            {
                "boxes": torch.tensor([[0.6222, 0.5537, 0.7987, 0.9782]]),
                "scores": torch.tensor([0.6955]),
                "labels": torch.tensor([69]),
            },
            {
                "boxes": torch.tensor([[0.1368, 0.5415, 0.2967, 0.7625]]),
                "scores": torch.tensor([0.8974]),
                "labels": torch.tensor([56]),
            },
        ]
        self.dummy_preds_4 = [
            {
                "boxes": torch.tensor(
                    [
                        [
                            0.5434,
                            0.2523,
                            0.7639,
                            0.2523,
                            0.7639,
                            0.7101,
                            0.5434,
                            0.7101,
                            0.0,
                            0.0,
                        ]
                    ]
                ),
                "scores": torch.tensor([0.9896]),
                "labels": torch.tensor([0]),
            },
            {
                "boxes": torch.tensor(
                    [
                        [
                            0.5795,
                            0.3360,
                            0.9880,
                            0.3360,
                            0.9880,
                            0.8807,
                            0.5795,
                            0.8807,
                        ],
                        [
                            0.3802,
                            0.7503,
                            0.6791,
                            0.7503,
                            0.6791,
                            0.8499,
                            0.3802,
                            0.8499,
                        ],
                    ]
                ),
                "scores": torch.tensor([0.9610, 0.3281]),
                "labels": torch.tensor([0, 43]),
            },
            {
                "boxes": torch.tensor(
                    [
                        [
                            2.5787e-01,
                            3.2486e-01,
                            5.9471e-01,
                            3.2486e-01,
                            5.9471e-01,
                            7.8843e-01,
                            2.5787e-01,
                            5.9471e-01,
                        ]
                    ]
                ),
                "scores": torch.tensor([0.9981]),
                "labels": torch.tensor([0]),
            },
            {
                "boxes": torch.tensor(
                    [[0.5187, 0.7525, 0.0, 0.0, 0.8490, 0.9353, 0.0, 0.0]]
                ),
                "scores": torch.tensor([0.9354]),
                "labels": torch.tensor([66]),
            },
            {
                "boxes": torch.tensor(
                    [
                        [
                            1.9219e-01,
                            2.4047e-01,
                            7.1588e-01,
                            2.4047e-01,
                            7.1588e-01,
                            9.7711e-01,
                            1.9219e-01,
                            9.7711e-01,
                        ]
                    ]
                ),
                "scores": torch.tensor([0.9872]),
                "labels": torch.tensor([0]),
            },
            {
                "boxes": torch.tensor(
                    [
                        [
                            0.6809,
                            0.2188,
                            0.8977,
                            0.2188,
                            0.8977,
                            0.8379,
                            0.6809,
                            0.8379,
                        ]
                    ]
                ),
                "scores": torch.tensor([0.9821]),
                "labels": torch.tensor([72]),
            },
            {
                "boxes": torch.tensor(
                    [
                        [
                            0.0055,
                            0.1611,
                            0.5952,
                            0.1611,
                            0.5952,
                            0.7966,
                            0.0055,
                            0.7966,
                        ]
                    ]
                ),
                "scores": torch.tensor([0.8929]),
                "labels": torch.tensor([0]),
            },
            {
                "boxes": torch.tensor(
                    [
                        [
                            0.3045,
                            0.5533,
                            0.4444,
                            0.5533,
                            0.4444,
                            0.6427,
                            0.3045,
                            0.6427,
                        ]
                    ]
                ),
                "scores": torch.tensor([0.5214]),
                "labels": torch.tensor([45]),
            },
            {
                "boxes": torch.tensor(
                    [
                        [
                            0.0216,
                            0.2478,
                            0.4436,
                            0.2478,
                            0.4436,
                            0.9219,
                            0.0216,
                            0.9219,
                        ]
                    ]
                ),
                "scores": torch.tensor([0.9635]),
                "labels": torch.tensor([0]),
            },
            {
                "boxes": torch.tensor(
                    [
                        [
                            0.0972,
                            0.2937,
                            0.4919,
                            0.2937,
                            0.4919,
                            0.7791,
                            0.0972,
                            0.7791,
                        ]
                    ]
                ),
                "scores": torch.tensor([0.9990]),
                "labels": torch.tensor([0]),
            },
            {
                "boxes": torch.tensor(
                    [
                        [
                            0.5352,
                            0.3373,
                            0.6268,
                            0.3373,
                            0.6268,
                            0.3783,
                            0.5352,
                            0.3783,
                        ]
                    ]
                ),
                "scores": torch.tensor([0.1972]),
                "labels": torch.tensor([73]),
            },
            {
                "boxes": torch.tensor(
                    [
                        [
                            0.1826,
                            0.3376,
                            0.3629,
                            0.3376,
                            0.3629,
                            0.4462,
                            0.1826,
                            0.4462,
                        ]
                    ]
                ),
                "scores": torch.tensor([0.8671]),
                "labels": torch.tensor([68]),
            },
            {
                "boxes": torch.tensor(
                    [
                        [
                            2.6635e-01,
                            6.4096e-01,
                            4.3022e-01,
                            6.4096e-01,
                            4.3022e-01,
                            7.6787e-01,
                            2.6635e-01,
                            7.6787e-01,
                        ]
                    ]
                ),
                "scores": torch.tensor([0.7603]),
                "labels": torch.tensor([61]),
            },
            {
                "boxes": torch.tensor(
                    [
                        [
                            0.4873,
                            0.5500,
                            0.6051,
                            0.5500,
                            0.6051,
                            0.8983,
                            0.4873,
                            0.8983,
                        ]
                    ]
                ),
                "scores": torch.tensor([0.8823]),
                "labels": torch.tensor([0]),
            },
            {
                "boxes": torch.tensor(
                    [
                        [
                            0.6222,
                            0.5537,
                            0.9782,
                            0.5537,
                            0.7987,
                            0.9782,
                            0.6222,
                            0.9782,
                        ]
                    ]
                ),
                "scores": torch.tensor([0.6955]),
                "labels": torch.tensor([69]),
            },
            {
                "boxes": torch.tensor(
                    [
                        [
                            0.1368,
                            0.5415,
                            0.2967,
                            0.5415,
                            0.2967,
                            0.7625,
                            0.1368,
                            0.7625,
                        ]
                    ]
                ),
                "scores": torch.tensor([0.8974]),
                "labels": torch.tensor([56]),
            },
        ]
        self.eps = 1e-6

    def test_coco_evaluation(self):
        ap50_95, ap50, ap_dict, summary = coco_evaluation(
            self.dummy_preds_coco, self.dataset, print_cls_ap=True
        )
        self.assertLessEqual(ap50_95, 1.0)
        self.assertGreaterEqual(ap50_95, 0.0)
        self.assertLessEqual(ap50, 1.0)
        self.assertGreaterEqual(ap50, 0.0)
        self.assertEqual(len(ap_dict), 160)
        self.assertEqual(type(summary), str)

    def test_convert_to_coco_format(self):
        res = convert_to_coco_format(self.dummy_preds, self.dataset)
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), 17)
        self.assertEqual(len(res[0]), 4)

    def test_convert_4pointBbox_to_coco_format(self):
        res = convert_to_coco_format(self.dummy_preds_4, self.dataset)
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), 17)
        self.assertEqual(len(res[0]), 4)
