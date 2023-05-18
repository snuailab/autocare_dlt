import torch.nn.functional as F
from torchvision.ops.feature_pyramid_network import (
    FeaturePyramidNetwork as TVFPN,
)
from torchvision.ops.feature_pyramid_network import (
    LastLevelMaxPool,
    LastLevelP6P7,
)


class FeaturePyramidNetwork(TVFPN):
    def __init__(self, in_channels, out_channels, extra_blocks=None):
        if not isinstance(out_channels, int):
            raise ValueError(f"out_channels: {out_channels} must be int value")
        if extra_blocks == "pool":
            extra_blocks = LastLevelMaxPool()
        else:
            extra_blocks = LastLevelP6P7(out_channels, out_channels)

        super().__init__(in_channels, out_channels, extra_blocks)

    def forward(self, x):
        if isinstance(x, dict):
            x = [f for f in x.values()]

        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        results = []
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(
                last_inner, size=feat_shape, mode="nearest"
            )
            last_inner = inner_lateral + inner_top_down
            results.insert(
                0, self.get_result_from_layer_blocks(last_inner, idx)
            )

        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, [])

        return results
