import torch
import torch.nn as nn

from autocare_dlt.core.model.utils.functions import make_divisible
from autocare_dlt.core.utils.boxes import cxcywh2xyxy


class YOLOv5Head(nn.Module):
    stride = None

    def __init__(
        self, 
        model_size, 
        num_classes, 
        anchors=None, 
        topk=1000 # topk is not used
    ):  # detection layer
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
        self.gw = gains[model_size.lower()]["gw"]

        ch = {"1": 256, "2": 512, "3": 1024}
        # ch_6 = {'1': 256, '2':512, '3':768, '4':1024}
        ch = self.re_channels_out(ch)
        ch = [c for c in ch.values()]
        if anchors is None:
            anchors = [
                [10, 13, 16, 30, 33, 23],
                [30, 61, 62, 45, 59, 119],
                [116, 90, 156, 198, 373, 326],
            ]
        else:
            anchors = anchors
        self.nc = nc = num_classes  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.num_layers = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.num_layers  # init grid
        self.anchor_grid = [
            torch.zeros(1)
        ] * self.num_layers  # init anchor grid
        a = torch.tensor(anchors).float().view(self.num_layers, -1, 2)
        self.register_buffer("anchors", a)  # shape(nl,na,2)
        self.m = nn.ModuleList(
            nn.Conv2d(x, self.no * self.na, 1) for x in ch
        )  # output conv
        self.stride = torch.tensor(
            [256.0 / x for x in [32.0, 16.0, 8.0]]
        )  # forward
        check_anchor_order(self)  # must be in pixel-space (not grid-space)
        self.anchors /= self.stride.view(-1, 1, 1)
        #self.topk = topk

    def get_width(self, n):
        return make_divisible(n * self.gw, 8)

    def re_channels_out(self, ch_dict):
        for k, v in ch_dict.items():
            ch_dict[k] = self.get_width(v)
        return ch_dict

    def forward(self, input, img_size=None, **kwargs):
        z = []
        y = []
        x = [0] * self.num_layers  # inference output
        for i in range(self.num_layers):
            x[i] = self.m[i](input[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = (
                x[i]
                .view(bs, self.na, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(
                        nx, ny, i
                    )

                y_ = x[i].sigmoid()
                if kwargs.get("feature_extract", False):
                    y.append(y_.view(bs, -1, self.no).cpu().detach())
                xy, wh, conf = y_.split(
                    (2, 2, self.nc + 1), 4
                )  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                y_ = torch.cat((xy, wh, conf), 4)
                z.append(y_.view(bs, -1, self.no))
        if self.training:
            return x
        else:
            outputs = torch.cat(z, 1)
            if kwargs.get("feature_extract", False):
                feats = torch.cat(y, 1)
                return outputs.cpu().detach(), feats

            return self.postprocess_detections(outputs, img_size)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(
            nx, device=d, dtype=t
        )
        yv, xv = torch.meshgrid(y, x, indexing="ij")
        grid = (
            torch.stack((xv, yv), 2).expand(shape) - 0.5
        )  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (
            (self.anchors[i] * self.stride[i])
            .view((1, self.na, 1, 1, 2))
            .expand(shape)
        )
        return grid, anchor_grid

    def postprocess_detections(self, outputs, img_size):
        # outputs: [bs, num_img, (x, y, w, h, obj, cls)]

        box = outputs[:, :, :4]
        logit = (
            outputs[:, :, 5:] * outputs[:, :, 4:5]
            if self.nc > 1
            else outputs[:, :, 4:5]
        )
        conf, label = logit.max(-1)

        # label = torch.zeros_like(conf)
        # box = torch.zeros(conf.shape[0], self.topk, 4)
        # for i in range(label.shape[0]): #why it is working?
        #     label[i] = label_[i][keep[i]]
        #     box[i] = cxcywh2xyxy(box_[i][keep[i]])
        box = cxcywh2xyxy(box, batch_input=True)
        x = box[:, :, 0::2] / img_size[1]
        y = box[:, :, 1::2] / img_size[0]
        batch_img_boxes = (
            torch.stack((x[:, :, 0], y[:, :, 0], x[:, :, 1], y[:, :, 1]), -1)
            .cpu()
            .detach()
        )
        batch_img_scores = conf.cpu().detach()
        batch_img_labels = label.int().cpu().detach()

        return batch_img_boxes, batch_img_scores, batch_img_labels


def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = (
        m.anchors.prod(-1).mean(-1).view(-1)
    )  # mean anchor area per output layer
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da and (da.sign() != ds.sign()):  # same order
        m.anchors[:] = m.anchors.flip(0)
