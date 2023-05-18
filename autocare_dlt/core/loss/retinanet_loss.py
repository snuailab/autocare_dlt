from typing import List

import torch
from torch import Tensor, nn
from torchvision.models.detection._utils import BoxCoder, Matcher
from torchvision.ops import sigmoid_focal_loss

from autocare_dlt.core.loss import IOUloss
from autocare_dlt.core.utils import bboxes_iou


class RetinaNetLoss(nn.Module):  # TODO: change name? -> DetLossWithoutBG
    def __init__(
        self,
        fg_iou_thresh=0.45,
        bg_iou_thresh=0.2,
        cls_loss_cfg={
            "alpha": 0.25,
            "gamma": 2,
        },
        bbox_loss_cfg={"loss_type": "giou", "reduction": "sum"},
    ):
        super().__init__()

        self.alpha = cls_loss_cfg.get("alpha", 0.25)
        self.gamma = cls_loss_cfg.get("gamma", 2)
        self.iou_loss = IOUloss(**bbox_loss_cfg)
        self.proposal_matcher = Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=True,
        )
        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.BETWEEN_THRESHOLDS = Matcher.BETWEEN_THRESHOLDS

    def forward(self, outputs, labels, **kwargs):
        matched_idxs = []
        cls_logits = outputs["cls_logits"]
        bbox_regression = outputs["bbox_regression"]
        anchors = outputs["anchors"]
        for an, l in zip(anchors, labels):
            if l["boxes"] is None:
                matched_idxs.append(
                    torch.full(
                        (an.size(0),), -1, dtype=torch.int64, device=an.device
                    )
                )
                continue
            img_ious = bboxes_iou(l["boxes"], an, xyxy=True)
            if len(img_ious) > 0:
                matched_idxs.append(self.proposal_matcher(img_ious))
        losses = self.compute_loss(
            labels, cls_logits, bbox_regression, anchors, matched_idxs
        )
        return losses

    def compute_loss(
        self,
        targets,
        cls_logits,
        bbox_regression,
        anchors,
        matched_idxs,
    ):
        bbox_losses = []
        cls_losses = []

        for (
            targets_per_image,
            cls_logits_per_image,
            bbox_regression_per_image,
            anchors_per_image,
            matched_idxs_per_image,
        ) in zip(targets, cls_logits, bbox_regression, anchors, matched_idxs):
            # determine only the foreground indices, ignore the rest
            foreground_idxs_per_image = torch.where(
                matched_idxs_per_image >= 0
            )[0]
            num_foreground = foreground_idxs_per_image.numel()
            gt_classes_target = torch.zeros_like(cls_logits_per_image)
            loss_bg = 0
            loss_fg = 0
            if num_foreground > 0:
                gt_classes_target[
                    foreground_idxs_per_image,
                    targets_per_image["labels"][
                        matched_idxs_per_image[foreground_idxs_per_image]
                    ],
                ] = 1.0
                loss = sigmoid_focal_loss(
                    cls_logits_per_image,
                    gt_classes_target,
                    alpha=self.alpha,
                    gamma=self.gamma,
                    reduction="none",
                )
                loss_fg = loss[foreground_idxs_per_image].sum()

                # compute the regression targets

                bg_idx = matched_idxs_per_image < 0
                valid_idxs_per_image = (
                    matched_idxs_per_image != self.BETWEEN_THRESHOLDS
                )
                valid_bg = valid_idxs_per_image == bg_idx
                loss_bg, _ = torch.sort(loss[valid_bg].sum(1), descending=True)
                loss_bg = loss_bg.sum()

                matched_gt_boxes_per_image = targets_per_image["boxes"][
                    matched_idxs_per_image[foreground_idxs_per_image]
                ]
                bbox_regression_per_image = bbox_regression_per_image[
                    foreground_idxs_per_image, :
                ]
                anchors_per_image = anchors_per_image[
                    foreground_idxs_per_image, :
                ]
                bbox_pred = self.box_coder.decode_single(
                    bbox_regression_per_image, anchors_per_image
                )
                bbox_losses.append(
                    self.iou_loss(bbox_pred, matched_gt_boxes_per_image)
                    / max(1, num_foreground)
                )

            cls_losses.append((loss_fg + loss_bg) / max(1, num_foreground))

        cls_loss = _sum(cls_losses) / max(1, len(targets))
        bbox_loss = _sum(bbox_losses) / max(1, len(targets)) * 2

        out = {
            "bbox_loss": bbox_loss,
            "cls_loss": cls_loss,
        }

        return out


def _sum(x: List[Tensor]) -> Tensor:
    if len(x) == 0:
        return 0
    res = x[0]
    for i in x[1:]:
        res = res + i
    return res
