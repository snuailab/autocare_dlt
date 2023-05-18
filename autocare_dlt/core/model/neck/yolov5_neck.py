#!/usr/bin/env python3
import torch.nn as nn

from autocare_dlt.core.model.utils.functions import make_divisible
from autocare_dlt.core.model.utils.yolov5_blocks import C3, Concat, Conv


class YOLOv5Neck(nn.Module):
    def __init__(
        self,
        model_size,
    ):
        super().__init__()
        if model_size not in ["n", "s", "m", "l", "x"]:
            raise ValueError(
                f"model_size: {model_size} should be in ['n', 's', 'm', 'l', 'x']"
            )
        gains = {
            "n": {"gd": 0.33, "gw": 0.25},
            "s": {"gd": 0.33, "gw": 0.5},
            "m": {"gd": 0.67, "gw": 0.75},
            "l": {"gd": 1, "gw": 1},
            "x": {"gd": 1.33, "gw": 1.25},
        }
        self.gd = gains[model_size.lower()]["gd"]  # depth gain
        self.gw = gains[model_size.lower()]["gw"]  # width gain

        in_channels = {"1": 256, "2": 512, "3": 512}
        in_channels = self.re_channels_out(in_channels)
        # FPN
        inner_p4, outer_p4 = 512, 256
        C3_size, C4_size, C5_size = (
            in_channels["1"],
            in_channels["2"],
            in_channels["3"],
        )

        self.channels_out = {
            "inner_p4": inner_p4,
            "outer_p4": outer_p4,
        }
        self.channels_out = self.re_channels_out(self.channels_out)
        self.concat = Concat()

        self.P5_upsampled = nn.Upsample(scale_factor=2, mode="nearest")

        self.P4_1 = C3(
            C5_size + C4_size,
            self.channels_out["inner_p4"],
            self.get_depth(3),
            False,
        )
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
        self.P4_2 = Conv(
            self.channels_out["inner_p4"], self.channels_out["outer_p4"], 1, 1
        )

        # PAN
        inner_p3, inner_p4, inner_p5 = 256, 512, 1024
        P3_size = C3_size + self.channels_out["outer_p4"]
        P4_size = self.channels_out["outer_p4"]
        P5_size = C5_size

        self.channels_out = {
            "inner_p3": inner_p3,
            "inner_p4": inner_p4,
            "inner_p5": inner_p5,
        }
        self.channels_out = self.re_channels_out(self.channels_out)

        self.P3_size = P3_size
        self.P4_size = P4_size
        self.P5_size = P5_size
        self.inner_p3 = self.channels_out["inner_p3"]
        self.inner_p4 = self.channels_out["inner_p4"]
        self.inner_p5 = self.channels_out["inner_p5"]
        self.P3 = C3(self.P3_size, self.inner_p3, self.get_depth(3), False)
        self.convP3 = Conv(self.inner_p3, self.inner_p3, 3, 2)
        self.P4 = C3(
            self.P4_size + self.inner_p3,
            self.inner_p4,
            self.get_depth(3),
            False,
        )
        self.convP4 = Conv(self.inner_p4, self.inner_p4, 3, 2)
        self.P5 = C3(
            self.inner_p4 + P5_size, self.inner_p5, self.get_depth(3), False
        )
        self.out_shape = (self.inner_p3, self.inner_p4, self.inner_p5)

    def get_depth(self, n):
        return max(round(n * self.gd), 1) if n > 1 else n

    def get_width(self, n):
        return make_divisible(n * self.gw, 8)

    def re_channels_out(self, ch_dict):
        for k, v in ch_dict.items():
            ch_dict[k] = self.get_width(v)
        return ch_dict

    def forward(self, inputs):
        C3, C4, C5 = inputs
        up5 = self.P5_upsampled(C5)
        concat1 = self.concat([up5, C4])
        p41 = self.P4_1(concat1)
        P4 = self.P4_2(p41)
        up4 = self.P4_upsampled(P4)
        P3 = self.concat([up4, C3])
        P5 = C5
        PP3 = self.P3(P3)
        convp3 = self.convP3(PP3)
        concat3_4 = self.concat([convp3, P4])
        PP4 = self.P4(concat3_4)
        convp4 = self.convP4(PP4)
        concat4_5 = self.concat([convp4, P5])
        PP5 = self.P5(concat4_5)
        return PP3, PP4, PP5


# class YOLOv5_6Neck(nn.Module):
#     def __init__(
#         self,
#         model_size,
#     ):
#         super().__init__()
#         gains = {'n': {'gd': 0.33, 'gw': 0.25},
#                  's': {'gd': 0.33, 'gw': 0.5},
#                  'm': {'gd': 0.67, 'gw': 0.75},
#                  'l': {'gd': 1, 'gw': 1},
#                  'x': {'gd': 1.33, 'gw': 1.25}}
#         self.gd = gains[model_size.lower()]['gd']  # depth gain
#         self.gw = gains[model_size.lower()]['gw']  # width gain

#         in_channels={'1': 256, '2':512, '3':768, '4':1024}
#         in_channels= self.re_channels_out(in_channels)
#         # FPN
#         inner_p4, outer_p4 = 512, 256
#         C3_size, C4_size, C5_size, C6_size = in_channels['1'], in_channels['2'], in_channels['3'], in_channels['4']

#         self.channels_out = {
#             'inner_p4': inner_p4,
#             'outer_p4': outer_p4,
#         }
#         self.channels_out = self.re_channels_out(self.channels_out)
#         self.concat = Concat()

#         self.P6_1 = Conv(C6_size, C5_size, 1, 1)
#         self.P6_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

#         self.P5_1 = C3(C5_size + C5_size, C5_size, self.get_depth(3), False)
#         self.P5_2 = Conv(C5_size, C4_size, 1, 1)
#         self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

#         self.P4_1 = C3(C4_size + C4_size, C4_size, self.get_depth(3), False)
#         self.P4_2 = Conv(C4_size, C3_size, 1, 1)
#         self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

#         # # PAN
#         # inner_p3, inner_p4, inner_p5 = 256, 512, 1024
#         # P3_size = C3_size + self.channels_out['outer_p4']
#         # P4_size = self.channels_out['outer_p4']
#         # P5_size = C5_size

#         # self.channels_out = {
#         #     'inner_p3': inner_p3,
#         #     'inner_p4': inner_p4,
#         #     'inner_p5': inner_p5
#         # }
#         # self.channels_out = self.re_channels_out(self.channels_out)

#         # self.P3_size = P3_size
#         # self.P4_size = P4_size
#         # self.P5_size = P5_size
#         # self.inner_p3 = self.channels_out['inner_p3']
#         # self.inner_p4 = self.channels_out['inner_p4']
#         # self.inner_p5 = self.channels_out['inner_p5']
#         self.P3 = C3(C3_size + C3_size, C3_size, self.get_depth(3), False)

#         self.convP3 = Conv(C3_size, C3_size, 3, 2)
#         self.P4 = C3(C3_size + C3_size, C4_size, self.get_depth(3), False)

#         self.convP4 = Conv(C4_size, C4_size, 3, 2)
#         self.P5 = C3(C4_size + C4_size, C5_size, self.get_depth(3), False)

#         self.convP5 = Conv(C5_size, C5_size, 3, 2)
#         self.P6 = C3(C5_size + C5_size, C6_size, self.get_depth(3), False)
#         self.out_shape = (C3_size, C4_size, C5_size, C6_size)

#     def get_depth(self, n):
#         return max(round(n * self.gd), 1) if n > 1 else n

#     def get_width(self, n):
#         return make_divisible(n * self.gw, 8)

#     def re_channels_out(self, ch_dict):
#         for k, v in ch_dict.items():
#             ch_dict[k] = self.get_width(v)
#         return ch_dict

#     def forward(self, inputs):
#         C3, C4, C5, C6 = inputs
#         P6 = self.P6_1(C6)
#         up6 = self.P6_upsampled(P6)
#         concat1 = self.concat([up6, C5])
#         p51 = self.P5_1(concat1)
#         P5 = self.P5_2(p51)

#         up5 = self.P5_upsampled(P5)
#         concat1 = self.concat([up5, C4])
#         p41 = self.P4_1(concat1)
#         P4 = self.P4_2(p41)

#         up4 = self.P4_upsampled(P4)
#         P3 = self.concat([C3, up4])
#         PP3 = self.P3(P3)

#         convp3 = self.convP3(PP3)
#         concat3_4 = self.concat([convp3, P4])
#         PP4 = self.P4(concat3_4)

#         convp4 = self.convP4(PP4)
#         concat4_5 = self.concat([convp4, P5])
#         PP5 = self.P5(concat4_5)

#         convp5 = self.convP5(PP5)
#         concat5_6 = self.concat([convp5, P6])
#         PP6 = self.P6(concat5_6)

#         return PP3, PP4, PP5, PP6
