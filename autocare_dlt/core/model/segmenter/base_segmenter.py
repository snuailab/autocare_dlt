import sys

import timm
import torch.nn as nn

from autocare_dlt.core.model.backbone import UNet
from autocare_dlt.core.model.head import SegmentationHead
from autocare_dlt.core.model.neck import Identity

CUSTOM_BACKBONES = ["UNet"]


class BaseSegmenter(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        backbone = model_cfg.pop("backbone", None)
        neck = model_cfg.pop("neck", None)
        head = model_cfg.pop("head", None)
        if head is not None:
            n_classes = head["num_classes"]
        model_size = model_cfg.pop("model_size", None)
        in_channels = model_cfg.pop("in_channels", 3)

        if model_size:
            backbone["model_size"] = model_size
            neck["model_size"] = model_size
            head["model_size"] = model_size
        
        if backbone is not None:
            backbone_name = backbone.pop("name", None)
            if backbone_name in CUSTOM_BACKBONES:
                backbone = getattr(sys.modules[__name__], backbone_name)(
                    in_channels=in_channels, n_classes=n_classes, **backbone
                )
            else:
                backbone = timm.create_model(
                    backbone_name,
                    features_only=False,
                    pretrained=True,
                    in_chans=in_channels,
                    num_classes=n_classes,
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

