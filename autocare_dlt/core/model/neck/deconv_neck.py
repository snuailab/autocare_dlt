from torch import nn

from autocare_dlt.core.model.utils.functions import xavier_init


class DeconvNeck(nn.Module):
    def __init__(self, in_channels=512, out_channels=256, num_layers=3):
        super().__init__()

        layers = []
        inplanes = in_channels
        if isinstance(out_channels, int):
            out_channels = [out_channels] * num_layers
        else:
            if len(out_channels) != num_layers:
                raise ValueError(
                    f"num of deconv layers: {num_layers} and num of out_channels: {out_channels} must be same"
                )

        for i in range(num_layers):
            planes = out_channels[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=inplanes,
                    out_channels=planes,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            inplanes = planes

        self.deconv_block = nn.Sequential(*layers)

        xavier_init(self.deconv_block)

    def forward(self, x):
        return self.deconv_block(x)
