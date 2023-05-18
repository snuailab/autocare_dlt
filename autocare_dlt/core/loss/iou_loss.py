import math

import numpy as np
import torch
import torch.nn as nn


class IOUloss(nn.Module):
    loss_types = ["iou", "giou", "ciou", "diou"]
    reductions = ["mean", "sum", "none"]

    def __init__(self, reduction="none", loss_type="iou", xyxy=True):
        super().__init__()
        if loss_type not in self.loss_types:
            raise KeyError(f"loss_type: {loss_type} is not supported.")
        if reduction not in self.reductions:
            raise KeyError(f"reduction: {reduction} is not supported.")
        self.reduction = reduction
        self.loss_type = loss_type
        self.xyxy = xyxy

    def forward(self, pred, target, return_iou=False):
        if pred.shape[0] != target.shape[0]:
            raise ValueError(
                f"the number of predections ({pred.shape[0]}) and the number of targets ({target.shape[0]}) are not the same."
            )
        eps = 1e-16
        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        if self.xyxy:  # x1, y1, x2, y2 = pred
            b1_x1, b1_y1, b1_x2, b1_y2 = pred.chunk(4, 1)
            b2_x1, b2_y1, b2_x2, b2_y2 = target.chunk(4, 1)
            w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
            w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

        else:  # transform from xywh to xyxy
            (x1, y1, w1, h1), (x2, y2, w2, h2) = pred.chunk(
                4, 1
            ), target.chunk(4, 1)
            w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
            b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
            b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(
            0
        ) * (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        union = w1 * h1 + w2 * h2 - inter + eps

        # IoU
        iou = inter / union
        if self.loss_type in ["giou", "ciou", "diou"]:
            cw = torch.max(b1_x2, b2_x2) - torch.min(
                b1_x1, b2_x1
            )  # convex (smallest enclosing box) width
            ch = torch.max(b1_y2, b2_y2) - torch.min(
                b1_y1, b2_y1
            )  # convex height
            if self.loss_type in [
                "ciou",
                "diou",
            ]:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
                c2 = cw**2 + ch**2 + eps  # convex diagonal squared
                rho2 = (
                    (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
                    + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
                ) / 4  # center dist ** 2
                if (
                    self.loss_type == "ciou"
                ):  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                    v = (4 / math.pi**2) * torch.pow(
                        torch.atan(w2 / h2) - torch.atan(w1 / h1), 2
                    )
                    with torch.no_grad():
                        alpha = v / (v - iou + (1 + eps))
                    iou = iou - (rho2 / c2 + v * alpha)  # CIoU
                else:
                    iou = iou - rho2 / c2  # DIoU
            else:
                c_area = cw * ch + eps  # convex area
                iou = (
                    iou - (c_area - union) / c_area
                )  # GIoU https://arxiv.org/pdf/1902.09630.pdf
        loss = 1 - iou
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        if return_iou:
            return loss, iou
        else:
            return loss
