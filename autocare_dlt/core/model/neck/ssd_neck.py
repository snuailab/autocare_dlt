import torch
from torch import nn
from torch.nn import functional as F

from autocare_dlt.core.model.utils.functions import xavier_init


class SSDNeck(nn.Module):
    def __init__(self, in_channels, out_channels, l2_norm_scale=20):
        super().__init__()
        if len(out_channels) < len(in_channels):
            raise ValueError(
                f"lenght of out_channels: {len(out_channels)} must be greater than or equal to "
                f"lenght of input_channels: {len(in_channels)}"
            )
        if in_channels != out_channels[: len(in_channels)]:
            raise ValueError(
                f"the first {len(in_channels)} channels of out_channels: {out_channels} "
                f"must be same as input_channels: {len(in_channels)}"
            )
        self.l2_norm_scale = l2_norm_scale
        if self.l2_norm_scale:
            self.scale_weight = nn.Parameter(
                torch.ones(in_channels[0]) * l2_norm_scale
            )

        extra_channels = out_channels[len(in_channels) - 1 :]
        self.extra_layers = nn.ModuleList()
        for i in range(len(extra_channels) - 1):
            ec_in = extra_channels[i]
            ec_out = extra_channels[i + 1]
            layer = nn.Sequential(
                nn.Conv2d(ec_in, ec_in // 4, kernel_size=1),
                nn.BatchNorm2d(ec_in // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    ec_in // 4, ec_out, kernel_size=3, padding=1, stride=2
                ),
                nn.BatchNorm2d(ec_out),
                nn.ReLU(inplace=True),
            )
            self.extra_layers.append(layer)
        xavier_init(self.extra_layers)

    def forward(self, out_features):
        if isinstance(out_features, dict):
            out_features = [f for f in out_features.values()]
        outs = out_features
        if self.l2_norm_scale:
            outs[0] = self.scale_weight.view(1, -1, 1, 1) * F.normalize(
                out_features[0]
            )
        feat = outs[-1]
        for layer in self.extra_layers:
            feat = layer(feat)
            outs.append(feat)
        return outs
