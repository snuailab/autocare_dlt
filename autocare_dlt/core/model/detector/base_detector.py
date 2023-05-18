import sys

import timm
import torch.nn as nn

from autocare_dlt.core.model.backbone import (
    VGG_16_backbone,
    YOLOv5Backbone,
)
from autocare_dlt.core.model.head import *
from autocare_dlt.core.model.neck import *

CUSTOM_BACKBONES = ["YOLOv5Backbone"]


class BaseDetector(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        backbone = model_cfg.pop("backbone", None)
        neck = model_cfg.pop("neck", None)
        head = model_cfg.pop("head", None)
        model_size = model_cfg.pop("model_size", None)
        if model_size:
            backbone["model_size"] = model_size
            neck["model_size"] = model_size
            head["model_size"] = model_size

        if backbone is not None:
            backbone_name = backbone.pop("name", None)
            if backbone_name in CUSTOM_BACKBONES:
                backbone = getattr(sys.modules[__name__], backbone_name)(
                    **backbone
                )
            else:
                backbone = timm.create_model(
                    backbone_name,
                    features_only=True,
                    pretrained=True,
                    **backbone
                )
        if neck is not None:
            neck_name = neck.pop("name", None)
            neck = getattr(sys.modules[__name__], neck_name)(**neck)
        if head is not None:
            head_name = head.pop("name", None)
            head = getattr(sys.modules[__name__], head_name)(**head)

        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self):
        pass
