import sys

import timm
import torch.nn as nn

from autocare_dlt.core.model.head import *
from autocare_dlt.core.model.neck import *


class BasePoseEstimation(nn.Module):
    def __init__(self, model_cfg):
        """

        When using hrnet, ``out_indices`` must be [1] (out of [0, 1, 2 ,3 ,4]) to be
        connected to the head. The ``out_indices`` represents the i-th
        stage's output(feature map) from all the output(feature maps
        from all the stages). (See the architecture at the paper
        (https://arxiv.org/pdf/1908.07919v2.pdf). The ``out_indecies``
        is defined in ``models/HumanPoseNet.json``.

        The hrnet in ``timm`` also has the cfg ``feature_location``.
        "icre" increases the num of channels. "" does nothing

        ``hrnet`` source code is
        https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/hrnet.py

        """

        super().__init__()

        self.model_cfg = model_cfg
        backbone = model_cfg.get("backbone", None)
        neck = model_cfg.get("neck", None)
        head = model_cfg.get("head", None)
        model_size = model_cfg.get("model_size", None)
        if model_size:
            backbone["model_size"] = model_size
            neck["model_size"] = model_size
            head["model_size"] = model_size

        if backbone.get("name") is not None:
            backbone_name = backbone.pop("name", None)
            backbone = timm.create_model(
                backbone_name,
                features_only=True,
                pretrained=True,
                **backbone,
            )
        if neck.get("name") is not None:
            neck_name = neck.pop("name", None)
            neck = getattr(sys.modules[__name__], neck_name)(**neck)

        if head.get("name") is not None:
            head_name = head.pop("name", None)
            head = getattr(sys.modules[__name__], head_name)(**head)

        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x):
        pass
