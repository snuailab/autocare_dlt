#!/usr/bin/env python
import math

import torch
import torch.nn as nn

from autocare_dlt.core.model.detector import BaseDetector


class YOLOv5(BaseDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stride = self.head.stride
        self._initialize_biases()
        initialize_weights(self)

    def _initialize_biases(self, cf=None):
        # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        m = self.head  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(
                8 / (640 / s) ** 2
            )  # obj (8 objects per 640 image)
            b.data[:, 5:] += (
                math.log(0.6 / (m.nc - 0.99))
                if cf is None
                else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x, **kwargs):
        img_size = x.shape[-2:]
        out_features = self.backbone(x)

        neck_outs = self.neck(out_features)

        outputs = self.head(list(neck_outs), img_size, **kwargs)

        return outputs


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True
