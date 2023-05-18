import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models.detection import _utils as det_utils

from autocare_dlt.core.loss import BCE_FocalLoss, IOUloss
from autocare_dlt.core.utils import bboxes_iou


class SSDLoss(nn.Module):  # TODO: change name? -> DetLossWithBG (or HardNeg?)
    def __init__(
        self,
        iou_thres=0.45,
        neg_to_pos_ratio=3,
        bbox_loss_cfg={"loss_type": "giou", "reduction": "sum"},
    ):
        super().__init__()

        self.iou_loss = IOUloss(**bbox_loss_cfg)
        self.proposal_matcher = det_utils.SSDMatcher(iou_thres)
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.neg_to_pos_ratio = neg_to_pos_ratio

    def forward(self, outputs, labels, **kwargs):
        matched_idxs = []
        cls_logits = outputs["cls_logits"]
        bbox_regression = outputs["bbox_regression"]
        anchors = outputs["anchors"]
        for an, l in zip(anchors, labels):
            if l["boxes"] is None:
                matched_idxs.append(torch.full((an.size(0),), -1))
                continue
            img_ious = bboxes_iou(l["boxes"], an, xyxy=True)
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

        # Match original targets with default boxes
        num_foreground = 0
        bbox_loss = []
        cls_targets = []
        num_classes = cls_logits.size(-1)
        for (
            targets_per_image,
            bbox_regression_per_image,
            cls_logits_per_image,
            anchors_per_image,
            matched_idxs_per_image,
        ) in zip(targets, bbox_regression, cls_logits, anchors, matched_idxs):
            # create BG class targets
            gt_classes_target = torch.ones(
                (cls_logits_per_image.size(0),),
                dtype=torch.long,
                device=cls_logits_per_image.device,
            ) * (num_classes - 1)
            # produce the matching between boxes and targets
            foreground_idxs_per_image = torch.where(
                matched_idxs_per_image >= 0
            )[0]
            if len(foreground_idxs_per_image) > 0:
                foreground_matched_idxs_per_image = matched_idxs_per_image[
                    foreground_idxs_per_image
                ]
                num_foreground += foreground_matched_idxs_per_image.numel()

                # Calculate regression loss
                matched_gt_boxes_per_image = targets_per_image["boxes"][
                    foreground_matched_idxs_per_image
                ]
                bbox_regression_per_image = bbox_regression_per_image[
                    foreground_idxs_per_image, :
                ]
                anchors_per_image = anchors_per_image[
                    foreground_idxs_per_image, :
                ]
                # target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
                bbox_pred = self.box_coder.decode_single(
                    bbox_regression_per_image, anchors_per_image
                )
                bbox_loss.append(
                    self.iou_loss(bbox_pred, matched_gt_boxes_per_image)
                )

                # Estimate ground truth for class targets
                gt_classes_target[
                    foreground_idxs_per_image
                ] = targets_per_image["labels"][
                    foreground_matched_idxs_per_image
                ]
            cls_targets.append(gt_classes_target)

        bbox_loss = torch.stack(bbox_loss)
        cls_targets = torch.stack(cls_targets)

        # Calculate classification loss
        cls_loss = F.cross_entropy(
            cls_logits.view(-1, num_classes),
            cls_targets.view(-1),
            reduction="none",
        ).view(cls_targets.size())

        # Hard Negative Sampling
        foreground_idxs = cls_targets < num_classes - 1
        num_negative = self.neg_to_pos_ratio * foreground_idxs.sum(
            1, keepdim=True
        )
        # num_negative[num_negative < self.neg_to_pos_ratio] = self.neg_to_pos_ratio
        negative_loss = cls_loss.clone()
        negative_loss[foreground_idxs] = -float(
            "inf"
        )  # use -inf to detect positive values that creeped in the sample
        values, idx = negative_loss.sort(1, descending=True)
        # background_idxs = torch.logical_and(idx.sort(1)[1] < num_negative, torch.isfinite(values))
        background_idxs = idx.sort(1)[1] < num_negative

        N = max(1, num_foreground)
        bbox_loss = 2.0 * bbox_loss.sum() / N

        cls_loss = (
            cls_loss[foreground_idxs].sum() + cls_loss[background_idxs].sum()
        ) / N
        out = {
            "bbox_loss": bbox_loss,
            "cls_loss": cls_loss,
        }
        return out
