import sys

import timm
import torch.nn as nn

from autocare_dlt.core.model.head import ClassificationHead
from autocare_dlt.core.model.neck import Identity


class BaseClassifier(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        self.model_cfg = model_cfg

        backbone = model_cfg.pop("backbone", None)
        neck = model_cfg.pop("neck", None)
        head = model_cfg.pop("head", None)

        if backbone is not None:
            backbone_name = backbone.pop("name", None)
            backbone = timm.create_model(
                backbone_name, features_only=False, pretrained=True, **backbone
            )
            backbone = nn.Sequential(*list(backbone.children())[:-1])
        if neck is not None:
            neck_name = neck.pop("name", None)
            neck = getattr(sys.modules[__name__], neck_name)(**neck)
        if head is not None:
            head_name = head.pop("name", None)
            head = getattr(sys.modules[__name__], head_name)(**head)

        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x):
        pass
