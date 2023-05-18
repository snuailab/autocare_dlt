import random
from typing import Tuple

import albumentations as A
import numpy as np
import torch


class ImageAugmentation:
    def __init__(self, img_augs, mode="detection"):
        self.mode = mode
        self.augs = []
        self.normalized = False
        for aug_name, aug_cfg in img_augs.items():
            if aug_name == "MixUp":
                continue
            if aug_name == "ImageNormalization":
                self.augs.append(self.get_normalize(**aug_cfg))
                continue
            self.augs.append(getattr(A, aug_name)(**aug_cfg))
        if mode == "detection":
            self.augmentation = A.Compose(
                self.augs, bbox_params=A.BboxParams(format="pascal_voc")
            )
        elif mode == "keypoint":
            self.augmentation = A.Compose(
                self.augs,
                keypoint_params=A.KeypointParams(
                    format="xy", remove_invisible=False
                ),
            )

        else:
            self.augmentation = A.Compose(self.augs)

    def transform(
        self, img, labels=np.array([])
    ) -> Tuple[np.ndarray, torch.Tensor]:
        if self.mode == "detection":
            if len(labels) > 0:
                if len(labels[0]) == 9:
                    labels = self.labels_4point_2album(labels)
                else:
                    labels = self.labels2album(labels)
            if torch.is_tensor(img):
                img = img.numpy()
            transformed = self.augmentation(image=img, bboxes=labels)
            labels = np.array(transformed["bboxes"])
            if len(labels) > 0:
                if len(labels[0]) == 9:
                    labels = self.album2labels_4point(labels)
                else:
                    labels = self.album2labels(labels)

        elif self.mode == "keypoint":
            if torch.is_tensor(img):
                img = img.numpy()
            """
            ``img`` pixel value range:
                0 <= ``img``(int) <= 255 before ``self.augmentation()``.
                0.0 <= ``img``(float) <= 1.0 after ``self.augmentation()``.
            """
            keypoints = labels

            haskeypoints_mask = np.full((len(keypoints), 1), True)
            for joint_id in range(len(labels)):
                if np.all(labels[joint_id] == 0):
                    haskeypoints_mask[joint_id] = False

            transformed = self.augmentation(
                image=img,
                keypoints=keypoints,
                cropping_bbox=random.choice(keypoints),
            )
            keypoints_transformed = np.array(transformed["keypoints"])

            """
            `albumentation` forces the unlabeled keypoints to be labeled.
            The `haskeypints_mask` is used for them to be unlabeled back.
            """
            labels = keypoints_transformed * haskeypoints_mask

        else:
            transformed = self.augmentation(image=img)

        img = transformed["image"]

        if self.mode == "detection":
            if len(labels) > 0:
                # TODO: temporary not using
                for label in labels:
                    if (label[3] > 1).item() and (label[4] > 1):
                        assert max(label) <= max(img.shape)
                        if len(label) == 9:
                            label[2::2] /= img.shape[
                                0
                            ]  # normalized height 0-1
                            label[1::2] /= img.shape[1]
                        else:
                            label[[2, 4]] /= img.shape[
                                0
                            ]  # normalized height 0-1
                            label[[1, 3]] /= img.shape[1]

        labels = self.labels2tensor(labels)
        return img, labels

    def labels2tensor(self, labels):
        labels_out = torch.from_numpy(labels)
        return labels_out

    def labels2album(self, labels):
        # invert labels: (class, xmin, ymin, xmax, ymax) => (xmin, ymin, xmax, ymax, class)
        labels_a = np.zeros_like(labels)
        labels_a[:, :4] = labels[:, 1:]
        labels_a[:, 4] = labels[:, 0]
        return labels_a

    def labels_4point_2album(self, labels):
        # invert labels: (class, x1, y1, x2, y2, x3, y3, x4, y4) => (x1, y1, x2, y2, x3, y3, x4, y4, class)
        labels_a = np.zeros_like(labels)
        labels_a[:, :8] = labels[:, 1:]
        labels_a[:, 8] = labels[:, 0]
        return labels_a

    def album2labels(self, labels_a):
        # invert labels: (x1, y1, x2, y2, x3, y3, x4, y4, class) => (class, x1, y1, x2, y2, x3, y3, x4, y4)
        labels = np.zeros_like(labels_a)
        labels[:, 0] = labels_a[:, -1]
        labels[:, 1:] = labels_a[:, :4]
        return labels

    def album2labels_4point(self, labels_a):
        # invert labels: (xmin, ymin, xmax, ymax, class) => (class, xmin, ymin, xmax, ymax)
        labels = np.zeros_like(labels_a)
        labels[:, 0] = labels_a[:, -1]
        labels[:, 1:] = labels_a[:, :-1]
        return labels

    def get_normalize(self, type="base", factor=None):
        if type == "base":
            return A.Normalize(
                mean=(0, 0, 0),
                std=(1, 1, 1),
                max_pixel_value=255,
                always_apply=True,
            )
        elif type == "imagenet":
            return A.Normalize(always_apply=True)
        elif type == "deepstream":
            assert factor, "factor should be defined for the deepstream type"
            return A.Normalize(
                std=(1, 1, 1), max_pixel_value=1 / factor, always_apply=True
            )
        raise ValueError("Unrecognizable type of normalize")
