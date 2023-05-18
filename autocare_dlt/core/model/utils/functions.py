import math

import torch
import torch.nn as nn

from autocare_dlt.core.utils import nms, xyxy2cxcywh


def is_parallel(model):
    """check if model is in parallel mode."""
    parallel_type = (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )
    return isinstance(model, parallel_type)


def xavier_init(conv: nn.Module):
    for layer in conv.modules():
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.0)


def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


class FourPointBoxCoder:
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000.0 / 16)):
        # type: (Tuple[float, float, float, float], float) -> None
        """
        Args:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)
        targets = self.encode_single(reference_boxes, proposals)
        return targets.split(boxes_per_image, 0)

    def encode_single(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Args:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes_4_point(reference_boxes, proposals, weights)

        return targets

    def decode(self, rel_codes, boxes):
        # type: (Tensor, List[Tensor]) -> Tensor
        assert isinstance(boxes, (list, tuple))
        assert isinstance(rel_codes, torch.Tensor)
        boxes_per_image = [b.size(0) for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)
        box_sum = 0
        for val in boxes_per_image:
            box_sum += val
        if box_sum > 0:
            rel_codes = rel_codes.reshape(box_sum, -1)
        pred_boxes = self.decode_single(rel_codes, concat_boxes)
        if box_sum > 0:
            pred_boxes = pred_boxes.reshape(box_sum, -1, 4)
        return pred_boxes

    def decode_single(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Args:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """

        boxes = boxes.to(rel_codes.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]

        wx1, wy1, wx2, wy2, wx3, wy3, wx4, wy4 = self.weights

        pred_x1 = (
            rel_codes[:, 0::8] / wx1 * widths[:, None] + boxes[:, 0][:, None]
        )
        pred_y1 = (
            rel_codes[:, 1::8] / wy1 * heights[:, None] + boxes[:, 1][:, None]
        )
        pred_x2 = (
            rel_codes[:, 2::8] / wx2 * widths[:, None] + boxes[:, 2][:, None]
        )
        pred_y2 = (
            rel_codes[:, 3::8] / wy2 * heights[:, None] + boxes[:, 1][:, None]
        )
        pred_x3 = (
            rel_codes[:, 4::8] / wx3 * widths[:, None] + boxes[:, 2][:, None]
        )
        pred_y3 = (
            rel_codes[:, 5::8] / wy3 * heights[:, None] + boxes[:, 3][:, None]
        )
        pred_x4 = (
            rel_codes[:, 6::8] / wx4 * widths[:, None] + boxes[:, 0][:, None]
        )
        pred_y4 = (
            rel_codes[:, 7::8] / wy4 * heights[:, None] + boxes[:, 3][:, None]
        )

        pred_boxes = torch.stack(
            (
                pred_x1,
                pred_y1,
                pred_x2,
                pred_y2,
                pred_x3,
                pred_y3,
                pred_x4,
                pred_y4,
            ),
            dim=2,
        ).flatten(1)

        return pred_boxes


def encode_boxes_4_point(reference_boxes, proposals, weights):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
    """
    Encode a set of proposals with respect to some
    reference boxes

    Args:
        reference_boxes (Tensor): reference boxes
        proposals (Tensor): boxes to be encoded
        weights (Tensor[4]): the weights for ``(x, y, w, h)``
    """

    # perform some unpacking to make it JIT-fusion friendly
    wx1 = weights[0]
    wy1 = weights[1]
    wx2 = weights[2]
    wy2 = weights[3]
    wx3 = weights[4]
    wy3 = weights[5]
    wx4 = weights[6]
    wy4 = weights[7]

    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 1].unsqueeze(1)
    proposals_x3 = proposals[:, 2].unsqueeze(1)
    proposals_y3 = proposals[:, 3].unsqueeze(1)
    proposals_x4 = proposals[:, 0].unsqueeze(1)
    proposals_y4 = proposals[:, 3].unsqueeze(1)

    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)
    reference_boxes_x3 = reference_boxes[:, 4].unsqueeze(1)
    reference_boxes_y3 = reference_boxes[:, 5].unsqueeze(1)
    reference_boxes_x4 = reference_boxes[:, 6].unsqueeze(1)
    reference_boxes_y4 = reference_boxes[:, 7].unsqueeze(1)

    # implementation starts here
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y3 - proposals_y2

    targets_dx1 = wx1 * (reference_boxes_x1 - proposals_x1) / ex_widths
    targets_dy1 = wy1 * (reference_boxes_y1 - proposals_y1) / ex_heights
    targets_dx2 = wx2 * (reference_boxes_x2 - proposals_x2) / ex_widths
    targets_dy2 = wy2 * (reference_boxes_y2 - proposals_y2) / ex_heights
    targets_dx3 = wx3 * (reference_boxes_x3 - proposals_x3) / ex_widths
    targets_dy3 = wy3 * (reference_boxes_y3 - proposals_y3) / ex_heights
    targets_dx4 = wx4 * (reference_boxes_x4 - proposals_x4) / ex_widths
    targets_dy4 = wy4 * (reference_boxes_y4 - proposals_y4) / ex_heights

    targets = torch.cat(
        (
            targets_dx1,
            targets_dy1,
            targets_dx2,
            targets_dy2,
            targets_dx3,
            targets_dy3,
            targets_dx4,
            targets_dy4,
        ),
        dim=1,
    )
    return targets
