import torch.nn as nn


class VGG_KOR_FeatureExtractor(nn.Module):
    """FeatureExtractor of CRNN (https://arxiv.org/pdf/1507.05717.pdf)"""

    def __init__(self, input_channel, output_channel=512):
        super().__init__()
        self.output_channel = [
            int(output_channel / 8),
            int(output_channel / 4),
            int(output_channel / 2),
            output_channel,
        ]  # [64, 128, 256, 512]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 64x40x120
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 128x20x60
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1),
            nn.ReLU(True),  # 256x20x60
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2)),  # 256x10x30
            nn.Conv2d(
                self.output_channel[2],
                self.output_channel[3],
                3,
                1,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(self.output_channel[3]),
            nn.ReLU(True),  # 512x10x30
            nn.Conv2d(
                self.output_channel[3],
                self.output_channel[3],
                3,
                1,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(self.output_channel[3]),
            nn.ReLU(True),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0),
            nn.ReLU(True),
        )  # 512x1x24

    def forward(self, input):
        return self.ConvNet(input)
