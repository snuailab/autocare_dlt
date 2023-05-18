#!/usr/bin/env python
from torch import nn

from autocare_dlt.core.model.utils.yolov5_blocks import (
    C3,
    SPPF,
    Conv,
    Focus,
)


class YOLOv5Backbone(nn.Module):
    def __init__(self, focus=True, model_size="L", with_C3TR=False):
        super().__init__()
        self.with_focus = focus
        self.with_c3tr = False  # Do not support this module
        gains = {
            "n": {"gd": 0.33, "gw": 0.25},
            "s": {"gd": 0.33, "gw": 0.5},
            "m": {"gd": 0.67, "gw": 0.75},
            "l": {"gd": 1, "gw": 1},
            "x": {"gd": 1.33, "gw": 1.25},
        }
        self.gd = gains[model_size.lower()]["gd"]  # depth gain
        self.gw = gains[model_size.lower()]["gw"]  # width gain

        self.channels_out = {
            "stage1": 64,
            "stage2_1": 128,
            "stage2_2": 128,
            "stage3_1": 256,
            "stage3_2": 256,
            "stage4_1": 512,
            "stage4_2": 512,
            "stage5_1": 1024,
            "stage5_2": 1024,
            "spp": 1024,
            "csp1": 1024,
            "conv1": 512,
        }
        self.re_channels_out()

        if self.with_focus:
            self.stage1 = Focus(3, self.channels_out["stage1"])
        else:
            self.stage1 = Conv(3, self.channels_out["stage1"], 6, 2, 2)

        # for latest yolov5, you can change BottleneckCSP to C3
        self.stage2_1 = Conv(
            self.channels_out["stage1"],
            self.channels_out["stage2_1"],
            k=3,
            s=2,
        )
        self.stage2_2 = C3(
            self.channels_out["stage2_1"],
            self.channels_out["stage2_2"],
            self.get_depth(3),
        )
        self.stage3_1 = Conv(
            self.channels_out["stage2_2"], self.channels_out["stage3_1"], 3, 2
        )
        self.stage3_2 = C3(
            self.channels_out["stage3_1"],
            self.channels_out["stage3_2"],
            self.get_depth(6),
        )
        self.stage4_1 = Conv(
            self.channels_out["stage3_2"], self.channels_out["stage4_1"], 3, 2
        )
        self.stage4_2 = C3(
            self.channels_out["stage4_1"],
            self.channels_out["stage4_2"],
            self.get_depth(9),
        )
        self.stage5_1 = Conv(
            self.channels_out["stage4_2"], self.channels_out["stage5_1"], 3, 2
        )
        self.stage5_2 = C3(
            self.channels_out["stage5_1"],
            self.channels_out["stage5_2"],
            self.get_depth(3),
        )
        self.spp = SPPF(
            self.channels_out["stage5_1"], self.channels_out["spp"], k=5
        )
        self.conv1 = Conv(
            self.channels_out["csp1"], self.channels_out["conv1"], 1, 1
        )
        self.out_shape = {
            "C3_size": self.channels_out["stage3_2"],
            "C4_size": self.channels_out["stage4_2"],
            "C5_size": self.channels_out["conv1"],
        }

    def forward(self, x):
        x1 = self.stage1(x)
        x21 = self.stage2_1(x1)
        x22 = self.stage2_2(x21)
        x31 = self.stage3_1(x22)
        c3 = self.stage3_2(x31)
        x41 = self.stage4_1(c3)
        c4 = self.stage4_2(x41)
        x51 = self.stage5_1(c4)
        x52 = self.stage5_2(x51)
        spp = self.spp(x52)
        c5 = self.conv1(spp)
        return c3, c4, c5

    def get_depth(self, n):
        return max(round(n * self.gd), 1) if n > 1 else n

    def get_width(self, n):
        # NOTE: top level import brings 'circular import error'
        from autocare_dlt.core.model.utils.functions import make_divisible

        return make_divisible(n * self.gw, 8)

    def re_channels_out(self):
        for k, v in self.channels_out.items():
            self.channels_out[k] = self.get_width(v)


# # CSPDarkNet53
# class YOLOv5_6Backbone(nn.Module):
#     def __init__(
#       self,
#       focus=True,
#       model_size='L',
#       with_C3TR=False
#       ):
#         super(YOLOv5_6Backbone, self).__init__()
#         self.with_focus = focus
#         self.with_c3tr = with_C3TR
#         gains = {'n': {'gd': 0.33, 'gw': 0.25},
#                  's': {'gd': 0.33, 'gw': 0.5},
#                  'm': {'gd': 0.67, 'gw': 0.75},
#                  'l': {'gd': 1, 'gw': 1},
#                  'x': {'gd': 1.33, 'gw': 1.25}}
#         self.gd = gains[model_size.lower()]['gd']  # depth gain
#         self.gw = gains[model_size.lower()]['gw']  # width gain

#         self.channels_out = {
#             'stage1': 64,
#             'stage2_1': 128,
#             'stage2_2': 128,
#             'stage3_1': 256,
#             'stage3_2': 256,
#             'stage4_1': 512,
#             'stage4_2': 512,
#             'stage5_1': 768,
#             'stage5_2': 768,
#             'stage6_1': 1024,
#             'stage6_2': 1024,
#             'spp': 1024,
#         }
#         self.re_channels_out()

#         if self.with_focus:
#             self.stage1 = Focus(3, self.channels_out['stage1'])
#         else:
#             self.stage1 = Conv(3, self.channels_out['stage1'], 6, 2, 2)

#         # for latest yolov5, you can change BottleneckCSP to C3
#         self.stage2_1 = Conv(self.channels_out['stage1'], self.channels_out['stage2_1'], k=3, s=2)
#         self.stage2_2 = C3(self.channels_out['stage2_1'], self.channels_out['stage2_2'], self.get_depth(3))
#         self.stage3_1 = Conv(self.channels_out['stage2_2'], self.channels_out['stage3_1'], 3, 2)
#         self.stage3_2 = C3(self.channels_out['stage3_1'], self.channels_out['stage3_2'], self.get_depth(6))
#         self.stage4_1 = Conv(self.channels_out['stage3_2'], self.channels_out['stage4_1'], 3, 2)
#         self.stage4_2 = C3(self.channels_out['stage4_1'], self.channels_out['stage4_2'], self.get_depth(9))
#         self.stage5_1 = Conv(self.channels_out['stage4_2'], self.channels_out['stage5_1'], 3, 2)
#         self.stage5_2 = C3(self.channels_out['stage5_1'], self.channels_out['stage5_2'], self.get_depth(3))
#         self.stage6_1 = Conv(self.channels_out['stage5_2'], self.channels_out['stage6_1'], 3, 2)
#         self.stage6_2 = C3(self.channels_out['stage6_1'], self.channels_out['stage6_2'], self.get_depth(3))
#         # self.spp = SPP(self.channels_out['stage5'], self.channels_out['spp'], [5, 9, 13])
#         self.spp = SPPF(self.channels_out['stage6_2'], self.channels_out['spp'], k=5)
#         # if self.with_c3tr:
#         #     self.c3tr = C3TR(self.channels_out['spp'], self.channels_out['csp1'], self.get_depth(3), False)
#         # else:
#         #     self.csp1 = C3(self.channels_out['spp'], self.channels_out['csp1'], self.get_depth(3), False)
#         # self.conv1 = Conv(self.channels_out['csp1'], self.channels_out['conv1'], 1, 1)
#         # self.out_shape = {'C3_size': self.channels_out['stage3_2'],
#         #                   'C4_size': self.channels_out['stage4_2'],
#         #                   'C5_size': self.channels_out['conv1']}

#     def forward(self, x):
#         x = self.stage1(x)
#         x21 = self.stage2_1(x)
#         x22 = self.stage2_2(x21)
#         x31 = self.stage3_1(x22)
#         c3 = self.stage3_2(x31)
#         x41 = self.stage4_1(c3)
#         c4 = self.stage4_2(x41)
#         x51 = self.stage5_1(c4)
#         c5 = self.stage5_2(x51)
#         x61 = self.stage6_1(c5)
#         x62 = self.stage6_2(x61)
#         c6 = self.spp(x62)
#         # if not self.with_c3tr:
#         #     csp1 = self.csp1(spp)
#         #     c5 = self.conv1(csp1)
#         # else:
#         #     c3tr = self.c3tr(spp)
#         return c3, c4, c5, c6

#     def get_depth(self, n):
#         return max(round(n * self.gd), 1) if n > 1 else n

#     def get_width(self, n):
#         return make_divisible(n * self.gw, 8)

#     def re_channels_out(self):
#         for k, v in self.channels_out.items():
#             self.channels_out[k] = self.get_width(v)
