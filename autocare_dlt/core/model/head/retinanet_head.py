import math
from typing import List

import torch
import torchvision
from torch import Tensor, nn
from torchvision.models.detection import _utils as det_utils


class RetinaNetHead(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        aspect_ratio=[0.5, 1.0, 2.0],
        anchor_size=[
            32,
            64,
            128,
            256,
            512,
        ],  # length: num of output freatures from FPN
        topk_candidates=1000,
    ):
        super().__init__()
        if not isinstance(in_channels, int):
            raise ValueError(
                f"in_channels: {in_channels} must be int value"
            )  # default value 256
        self.num_classes = num_classes
        anchor_sizes = tuple(
            (x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3)))
            for x in anchor_size
        )
        aspect_ratios = (aspect_ratio,) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(
            anchor_sizes, aspect_ratios, xyxy=True
        )
        self.anchor_generator = anchor_generator
        num_anchors = anchor_generator.num_anchors_per_location()[0]
        self.classification_head = RetinaNetClassificationHead(
            in_channels, num_anchors, self.num_classes
        )
        self.regression_head = RetinaNetRegressionHead(
            in_channels, num_anchors
        )
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.topk_candidates = topk_candidates

    def forward(self, features, img_size, **kwargs):
        bbox_regression = self.regression_head(features)
        cls_logits = self.classification_head(features)
        anchors = self.anchor_generator(img_size, features)
        if self.training:
            return {
                "cls_logits": cls_logits,
                "bbox_regression": bbox_regression,
                "anchors": anchors,
            }
        else:
            # recover level sizes
            num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
            HW = 0
            for v in num_anchors_per_level:
                HW += v
            HWA = cls_logits.size(1)
            A = HWA // HW
            num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

            # split outputs per level
            split_cls_logits = list(
                cls_logits.split(num_anchors_per_level, dim=1)
            )
            split_bbox_regression = list(
                bbox_regression.split(num_anchors_per_level, dim=1)
            )
            split_anchors = [
                list(a.split(num_anchors_per_level)) for a in anchors
            ]
            if kwargs.get("feature_extract", False):
                out = self.detection_features(
                    split_bbox_regression, split_cls_logits, split_anchors
                )
            else:
                out = self.postprocess_detections(
                    split_bbox_regression, split_cls_logits, split_anchors
                )
        return out

    def postprocess_detections(self, bbox_regression, cls_logits, anchors):

        num_images = len(bbox_regression[0])
        batch_img_boxes = []
        batch_img_scores = []
        batch_img_labels = []

        for index in range(num_images):
            box_regression_per_image = [br[index] for br in bbox_regression]
            logits_per_image = [cl[index] for cl in cls_logits]
            anchors_per_image = anchors[index]
            image_boxes = []
            image_scores = []
            image_labels = []

            for (
                box_regression_per_level,
                logits_per_level,
                anchors_per_level,
            ) in zip(
                box_regression_per_image, logits_per_image, anchors_per_image
            ):
                num_classes = logits_per_level.shape[-1]
                scores_per_level = torch.sigmoid(logits_per_level).flatten()

                # keep only topk scoring predictions
                scores_per_level, topk_idxs = scores_per_level.topk(
                    self.topk_candidates
                )

                anchor_idxs = torch.floor(
                    torch.div(topk_idxs, num_classes)
                ).type(torch.int64)
                labels_per_level = topk_idxs % num_classes

                boxes_per_level = self.box_coder.decode_single(
                    box_regression_per_level[anchor_idxs],
                    anchors_per_level[anchor_idxs],
                )
                boxes_per_level = torchvision.ops.clip_boxes_to_image(
                    boxes_per_level, (1, 1)
                )

                image_boxes.append(boxes_per_level)
                image_scores.append(scores_per_level)
                image_labels.append(labels_per_level)

            batch_img_boxes.append(torch.cat(image_boxes, dim=0))
            batch_img_scores.append(torch.cat(image_scores, dim=0))
            batch_img_labels.append(torch.cat(image_labels, dim=0))

        batch_img_boxes = torch.stack(batch_img_boxes).cpu().detach()
        batch_img_scores = torch.stack(batch_img_scores).cpu().detach()
        batch_img_labels = torch.stack(batch_img_labels).cpu().detach()

        return batch_img_boxes, batch_img_scores, batch_img_labels

    def detection_features(self, bbox_regression, cls_logits, anchors):

        num_images = len(bbox_regression[0])
        batch_img_boxes = []
        batch_img_preds = []

        for index in range(num_images):
            box_regression_per_image = [br[index] for br in bbox_regression]
            logits_per_image = [cl[index] for cl in cls_logits]
            anchors_per_image = anchors[index]
            image_boxes = []
            image_preds = []

            for (
                box_regression_per_level,
                logits_per_level,
                anchors_per_level,
            ) in zip(
                box_regression_per_image, logits_per_image, anchors_per_image
            ):
                preds_per_level = torch.sigmoid(logits_per_level)

                # keep only topk scoring predictions

                boxes_per_level = self.box_coder.decode_single(
                    box_regression_per_level, anchors_per_level
                )
                boxes_per_level = torchvision.ops.clip_boxes_to_image(
                    boxes_per_level, (1, 1)
                )

                image_boxes.append(boxes_per_level)
                image_preds.append(preds_per_level)

            batch_img_boxes.append(torch.cat(image_boxes, dim=0))
            batch_img_preds.append(torch.cat(image_preds, dim=0))

        batch_img_boxes = torch.stack(batch_img_boxes).cpu().detach()
        batch_img_preds = torch.stack(batch_img_preds).cpu().detach()

        return batch_img_boxes, batch_img_preds


class RetinaNetClassificationHead(nn.Module):
    """
    A classification head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
    """

    def __init__(
        self, in_channels, num_anchors, num_classes, prior_probability=0.01
    ):
        super().__init__()

        conv = []
        for _ in range(4):
            conv.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            conv.append(nn.GroupNorm(32, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

        self.cls_logits = nn.Conv2d(
            in_channels,
            num_anchors * num_classes,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        torch.nn.init.constant_(
            self.cls_logits.bias,
            -math.log((1 - prior_probability) / prior_probability),
        )

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # This is to fix using det_utils.Matcher.BETWEEN_THRESHOLDS in TorchScript.
        # TorchScript doesn't support class attributes.

    def forward(self, x):
        # type: (List[Tensor]) -> Tensor
        all_cls_logits = []

        for features in x:
            cls_logits = self.conv(features)
            cls_logits = self.cls_logits(cls_logits)

            # Permute classification output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = cls_logits.shape
            cls_logits = cls_logits.view(N, -1, self.num_classes, H, W)
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
            cls_logits = cls_logits.reshape(
                N, -1, self.num_classes
            )  # Size=(N, HWA, 4)

            all_cls_logits.append(cls_logits)

        return torch.cat(all_cls_logits, dim=1)


class RetinaNetRegressionHead(nn.Module):
    """
    A regression head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """

    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
    }

    def __init__(self, in_channels, num_anchors):
        super().__init__()

        conv = []
        for _ in range(4):
            conv.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            conv.append(nn.GroupNorm(32, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        self.bbox_reg = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.normal_(self.bbox_reg.weight, std=0.01)
        torch.nn.init.zeros_(self.bbox_reg.bias)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        all_bbox_regression = []

        for features in x:
            bbox_regression = self.conv(features)
            bbox_regression = self.bbox_reg(bbox_regression)

            # Permute bbox regression output from (N, 4 * A, H, W) to (N, HWA, 4).
            N, _, H, W = bbox_regression.shape
            bbox_regression = bbox_regression.view(N, -1, 4, H, W)
            bbox_regression = bbox_regression.permute(0, 3, 4, 1, 2)
            bbox_regression = bbox_regression.reshape(
                N, -1, 4
            )  # Size=(N, HWA, 4)

            all_bbox_regression.append(bbox_regression)

        return torch.cat(all_bbox_regression, dim=1)


class AnchorGenerator(nn.Module):
    """
    Module that generates anchors for a set of feature maps and
    image sizes.

    The module support computing anchors at multiple sizes and aspect ratios
    per feature map. This module assumes aspect ratio = height / width for
    each anchor.

    sizes and aspect_ratios should have the same number of elements, and it should
    correspond to the number of feature maps.

    sizes[i] and aspect_ratios[i] can have an arbitrary number of elements,
    and AnchorGenerator will output a set of sizes[i] * aspect_ratios[i] anchors
    per spatial location for feature map i.

    Args:
        sizes (Tuple[Tuple[int]]):
        aspect_ratios (Tuple[Tuple[float]]):
    """

    __annotations__ = {
        "cell_anchors": List[torch.Tensor],
    }

    def __init__(
        self,
        sizes=((128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),),
        xyxy=True,
    ):
        super().__init__()

        if not isinstance(sizes[0], (list, tuple)):
            # TODO change this
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)

        assert len(sizes) == len(aspect_ratios)

        self.xyxy = xyxy
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = [
            self.generate_anchors(size, aspect_ratio)
            for size, aspect_ratio in zip(sizes, aspect_ratios)
        ]

    def generate_anchors(
        self,
        scales: List[int],
        aspect_ratios: List[float],
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(
            aspect_ratios, dtype=dtype, device=device
        )
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        if self.xyxy:
            base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        else:
            base_anchors = (
                torch.stack(
                    [torch.zeros_like(ws), torch.zeros_like(ws), ws, hs], dim=1
                )
                / 2
            )
        return base_anchors.round()

    def set_cell_anchors(self, dtype: torch.dtype, device: torch.device):
        self.cell_anchors = [
            cell_anchor.to(dtype=dtype, device=device)
            for cell_anchor in self.cell_anchors
        ]

    def num_anchors_per_location(self):
        return [
            len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)
        ]

    # For every combination of (a, (g, s), i) in (self.cell_anchors, zip(grid_sizes, strides), 0:2),
    # output g[i] anchors that are s[i] distance apart in direction i, with the same dimensions as a.
    def grid_anchors(self, grid_sizes, strides):
        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None

        if not (len(grid_sizes) == len(strides) == len(cell_anchors)):
            raise ValueError(
                "Anchors should be Tuple[Tuple[int]] because each feature "
                "map could potentially have different sizes and aspect ratios. "
                "There needs to be a match between the number of "
                "feature maps passed and the number of sizes / aspect ratios specified."
            )

        for size, stride, base_anchors in zip(
            grid_sizes, strides, cell_anchors
        ):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            # For output anchor, compute [x_center, y_center, x_center, y_center]
            shifts_x = (
                torch.arange(0, grid_width, dtype=torch.int64, device=device)
                * stride_width
            )
            shifts_y = (
                torch.arange(0, grid_height, dtype=torch.int64, device=device)
                * stride_height
            )
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            if self.xyxy:
                shifts = torch.stack(
                    (shift_x, shift_y, shift_x, shift_y), dim=1
                )
            else:
                shifts = torch.stack(
                    (
                        shift_x,
                        shift_y,
                        torch.zeros_like(shift_x),
                        torch.zeros_like(shift_x),
                    ),
                    dim=1,
                )

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(
                    -1, 4
                )
            )

        return anchors

    def forward(self, image_size, feature_maps):
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        strides = [
            [
                torch.tensor(
                    image_size[0] // g[0], dtype=torch.int64, device=device
                ),
                torch.tensor(
                    image_size[1] // g[1], dtype=torch.int64, device=device
                ),
            ]
            for g in grid_sizes
        ]
        self.set_cell_anchors(dtype, device)
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)
        for n in anchors_over_all_feature_maps:
            # Normalize anchors
            n[:, 0::2] /= image_size[0]
            n[:, 1::2] /= image_size[1]
        anchors: List[List[torch.Tensor]] = []
        for _ in range(len(feature_maps[0])):
            anchors_in_image = [
                anchors_per_feature_map
                for anchors_per_feature_map in anchors_over_all_feature_maps
            ]
            anchors.append(anchors_in_image)
        anchors = [
            torch.cat(anchors_per_image) for anchors_per_image in anchors
        ]
        return anchors
