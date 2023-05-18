import math
from typing import List, Optional

import torch
import torchvision
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.ssd import (
    SSDClassificationHead,
    SSDRegressionHead,
)


class SSDHead(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        aspect_ratio=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
        topk_candidates=400,
    ):
        super().__init__()
        self.num_classes = num_classes + 1
        self.anchor_generator = DefaultBoxGenerator(
            aspect_ratio, scales=scales
        )
        self.num_anchors = self.anchor_generator.num_anchors_per_location()
        self.classification_head = SSDClassificationHead(
            in_channels, self.num_anchors, self.num_classes
        )
        self.regression_head = SSDRegressionHead(in_channels, self.num_anchors)
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.topk_candidates = topk_candidates

    def forward(self, features, img_size, **kwargs):
        bbox_regression = self.regression_head(features)
        cls_logits = self.classification_head(features)
        anchors = self.anchor_generator(features, img_size, xyxy=True)
        if self.training:
            return {
                "cls_logits": cls_logits,
                "bbox_regression": bbox_regression,
                "anchors": anchors,
            }
        else:
            if kwargs.get("feature_extract", False):
                out = self.detection_features(
                    bbox_regression, cls_logits, anchors
                )
            else:
                out = self.postprocess_detections(
                    bbox_regression, cls_logits, anchors
                )
        return out

    def postprocess_detections(
        self, bbox_regression, cls_logits, image_anchors
    ):
        pred_scores = F.softmax(cls_logits, dim=-1)[:, :, :-1]
        image_anchors = image_anchors

        num_classes = pred_scores.size(-1)
        num_batch = pred_scores.shape[0]
        device = pred_scores.device

        batch_img_boxes = []
        batch_img_scores = []
        batch_img_labels = []
        for i in range(num_batch):
            boxes, scores, anchors = (
                bbox_regression[i],
                pred_scores[i],
                image_anchors[i],
            )
            boxes = self.box_coder.decode_single(boxes, anchors)
            if boxes.shape[-1] == 4:
                boxes = torchvision.ops.clip_boxes_to_image(boxes, (1, 1))

            image_boxes = []
            image_scores = []
            image_labels = []
            for label in range(num_classes):
                score = scores[:, label]

                """
                TensorRT does not support dynamic outputs!!
                option1: implement custom layer
                option2: set nvds-infer properties
                """
                # keep_idxs = torch.where(score> self.score_thresh)
                # score = score[keep_idxs]
                # box = boxes[keep_idxs]

                # keep only topk scoring predictions
                score, idxs = score.topk(self.topk_candidates)
                box = boxes[idxs]

                image_boxes.append(box)
                image_scores.append(score)
                image_labels.append(
                    torch.full_like(
                        score,
                        fill_value=label,
                        dtype=torch.long,
                        device=device,
                    )
                )

            batch_img_boxes.append(torch.cat(image_boxes, dim=0))
            batch_img_scores.append(torch.cat(image_scores, dim=0))
            batch_img_labels.append(torch.cat(image_labels, dim=0))

        batch_img_boxes = torch.stack(batch_img_boxes).cpu().detach()
        batch_img_scores = torch.stack(batch_img_scores).cpu().detach()
        batch_img_labels = torch.stack(batch_img_labels).cpu().detach()
        return batch_img_boxes, batch_img_scores, batch_img_labels

    def detection_features(self, bbox_regression, cls_logits, image_anchors):
        pred_scores = F.softmax(cls_logits, dim=-1)[:, :, :-1]

        num_classes = pred_scores.size(-1)

        batch_img_boxes = []
        batch_img_scores = []
        for boxes, scores, anchors in zip(
            bbox_regression, pred_scores, image_anchors
        ):
            boxes = self.box_coder.decode_single(boxes, anchors)
            if boxes.shape[-1] == 4:
                boxes = torchvision.ops.clip_boxes_to_image(boxes, (1, 1))

            batch_img_boxes.append(boxes)
            batch_img_scores.append(scores)

        batch_img_boxes = torch.stack(batch_img_boxes).cpu().detach()
        batch_img_scores = torch.stack(batch_img_scores).cpu().detach()
        return batch_img_boxes, batch_img_scores


class DefaultBoxGenerator(nn.Module):
    """
    This module generates the default boxes of SSD for a set of feature maps and image sizes.

    Args:
        aspect_ratios (List[List[int]]): A list with all the aspect ratios used in each feature map.
        min_ratio (float): The minimum scale :math:`\text{s}_{\text{min}}` of the default boxes used in the estimation
            of the scales of each feature map. It is used only if the ``scales`` parameter is not provided.
        max_ratio (float): The maximum scale :math:`\text{s}_{\text{max}}`  of the default boxes used in the estimation
            of the scales of each feature map. It is used only if the ``scales`` parameter is not provided.
        scales (List[float]], optional): The scales of the default boxes. If not provided it will be estimated using
            the ``min_ratio`` and ``max_ratio`` parameters.
        steps (List[int]], optional): It's a hyper-parameter that affects the tiling of defalt boxes. If not provided
            it will be estimated from the data.
        clip (bool): Whether the standardized values of default boxes should be clipped between 0 and 1. The clipping
            is applied while the boxes are encoded in format ``(cx, cy, w, h)``.
    """

    def __init__(
        self,
        aspect_ratios: List[List[int]],
        min_ratio: float = 0.15,
        max_ratio: float = 0.9,
        scales: Optional[List[float]] = None,
        steps: Optional[List[int]] = None,
        clip: bool = True,
    ):
        super().__init__()
        if steps is not None:
            assert len(aspect_ratios) == len(steps)
        self.aspect_ratios = aspect_ratios
        self.steps = steps
        self.clip = clip
        num_outputs = len(aspect_ratios)

        # Estimation of default boxes scales
        if scales is None:
            if num_outputs > 1:
                range_ratio = max_ratio - min_ratio
                self.scales = [
                    min_ratio + range_ratio * k / (num_outputs - 1.0)
                    for k in range(num_outputs)
                ]
                self.scales.append(1.0)
            else:
                self.scales = [min_ratio, max_ratio]
        else:
            self.scales = scales

        self._wh_pairs = self._generate_wh_pairs(num_outputs)

    def _generate_wh_pairs(
        self,
        num_outputs: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> List[Tensor]:
        _wh_pairs: List[Tensor] = []
        for k in range(num_outputs):
            # Adding the 2 default width-height pairs for aspect ratio 1 and scale s'k
            s_k = self.scales[k]
            s_prime_k = math.sqrt(self.scales[k] * self.scales[k + 1])
            wh_pairs = [[s_k, s_k], [s_prime_k, s_prime_k]]

            # Adding 2 pairs for each aspect ratio of the feature map k
            for ar in self.aspect_ratios[k]:
                sq_ar = math.sqrt(ar)
                w = self.scales[k] * sq_ar
                h = self.scales[k] / sq_ar
                wh_pairs.extend([[w, h]])

            _wh_pairs.append(
                torch.as_tensor(wh_pairs, dtype=dtype, device=device)
            )
        return _wh_pairs

    def num_anchors_per_location(self):
        # Estimate num of anchors based on aspect ratios: 2 default boxes + 2 * ratios of feaure map.
        return [2 + len(r) for r in self.aspect_ratios]

    # Default Boxes calculation based on page 6 of SSD paper
    def _grid_default_boxes(
        self,
        grid_sizes: List[List[int]],
        image_size: List[int],
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        default_boxes = []
        for k, f_k in enumerate(grid_sizes):
            # Now add the default boxes for each width-height pair
            if self.steps is not None:
                x_f_k, y_f_k = (
                    img_shape / self.steps[k] for img_shape in image_size
                )
            else:
                y_f_k, x_f_k = f_k

            shifts_x = ((torch.arange(0, f_k[1]) + 0.5) / x_f_k).to(
                dtype=dtype
            )
            shifts_y = ((torch.arange(0, f_k[0]) + 0.5) / y_f_k).to(
                dtype=dtype
            )
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            shifts = torch.stack(
                (shift_x, shift_y) * len(self._wh_pairs[k]), dim=-1
            ).reshape(-1, 2)
            # Clipping the default boxes while the boxes are encoded in format (cx, cy, w, h)
            _wh_pair = (
                self._wh_pairs[k].clamp(min=0, max=1)
                if self.clip
                else self._wh_pairs[k]
            )
            wh_pairs = _wh_pair.repeat((f_k[0] * f_k[1]), 1)

            default_box = torch.cat((shifts, wh_pairs), dim=1)

            default_boxes.append(default_box)

        return torch.cat(default_boxes, dim=0)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "aspect_ratios={aspect_ratios}"
        s += ", clip={clip}"
        s += ", scales={scales}"
        s += ", steps={steps}"
        s += ")"
        return s.format(**self.__dict__)

    def forward(self, feature_maps, image_size, xyxy=True):
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        default_boxes = self._grid_default_boxes(
            grid_sizes, image_size, dtype=dtype
        )
        default_boxes = default_boxes.to(device)

        dboxes = []
        for _ in range(len(feature_maps[0])):
            dboxes_in_image = default_boxes
            if xyxy:
                dboxes_in_image = torch.cat(
                    [
                        dboxes_in_image[:, :2] - 0.5 * dboxes_in_image[:, 2:],
                        dboxes_in_image[:, :2] + 0.5 * dboxes_in_image[:, 2:],
                    ],
                    -1,
                )
                # dboxes_in_image[:, 0::2] *= image_size[1]
                # dboxes_in_image[:, 1::2] *= image_size[0]
            dboxes.append(dboxes_in_image)
        return dboxes
