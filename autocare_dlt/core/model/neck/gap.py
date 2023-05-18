import torch.nn.functional as F
from torch import nn


class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features):

        results = []
        for feature in features:
            gap = F.adaptive_avg_pool2d(feature, (1, 1))
            results.append(gap.squeeze(-1).squeeze(-1))

        return results
