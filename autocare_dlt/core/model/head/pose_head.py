from torch import nn


class PoseHead(nn.Module):
    def __init__(self, in_channels, num_classes, **kargs):
        super().__init__()

        if not isinstance(in_channels, int):
            raise ValueError(f"in_channels: {in_channels} must be int value")

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.head = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, features):

        out = self.head(features)

        if self.training:
            return out
        else:
            return out.cpu().detach()
