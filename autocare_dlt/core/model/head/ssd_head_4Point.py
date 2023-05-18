from typing import List

from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models.detection.ssd import SSDScoringHead, _xavier_init

from autocare_dlt.core.model.head import SSDHead
from autocare_dlt.core.model.utils.functions import FourPointBoxCoder


class SSDRegressionHead_4Point(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int]):
        bbox_reg = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            bbox_reg.append(
                nn.Conv2d(channels, 8 * anchors, kernel_size=3, padding=1)
            )
        _xavier_init(bbox_reg)
        super().__init__(bbox_reg, 8)


class SSDHead4Point(SSDHead):
    def __init__(
        self,
        in_channels,
        num_classes,
        aspect_ratio=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
        topk_candidates=400,
    ):
        super().__init__(
            in_channels, num_classes, aspect_ratio, scales, topk_candidates
        )
        self.regression_head = SSDRegressionHead_4Point(
            in_channels, self.num_anchors
        )
        self.box_coder = FourPointBoxCoder(
            weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        )
