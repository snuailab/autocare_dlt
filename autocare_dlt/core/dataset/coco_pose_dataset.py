import copy
import logging
import os
from typing import Tuple

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from autocare_dlt.core.dataset.utils import (
    ImageAugmentation,
    img2tensor,
    letterbox,
)

logger = logging.getLogger(__name__)


class COCOPoseDataset(Dataset):
    def __init__(self, cfg, task_cfg):
        """
        Args:
            cfg (_type_): data_cfg['data'] + alpha(configured at the terminal input
            and "utils/config.py")
            task_cfg (_type_): data_cfg['data']['train'/'val'/'test']
        """

        # === Img cfgs === #
        self.img_size = cfg.get("img_size")
        if (not isinstance(self.img_size, list)) and (
            not isinstance(self.img_size, tuple)
        ):
            raise ValueError(
                f"data.img_size: {self.img_size} shuold be list of integer"
            )

        self.img_size = (
            [self.img_size[0], self.img_size[0]]
            if len(self.img_size) == 1
            else self.img_size
        )
        self.letter_box = cfg.get("letter_box", False)
        self.classes = cfg.get("classes")

        # === Joints cfgs === #
        # joint idx by keypoints ann file
        self.joints_dicts = self._get_joint_dict()
        self.num_joints = len(self.joints_dicts)

        # === Target cfgs === #
        self.heatmap_size = [
            int(self.img_size[0] / 4),
            int(self.img_size[1] / 4),
        ]
        # NOTE: hardcoded for now because it's configured in the model json
        self.sigma = 3
        size = 2 * (self.sigma * 3) + 1
        x = torch.arange(0, size, 1)
        y = x.unsqueeze(1)
        x0 = y0 = size // 2
        self.g = np.exp(
            -((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma**2)
        )

        # === Visualization cfgs === #
        self.in_vis_thre = 0.2
        self.oks_thre = 0.9

        # === Path cfgs === #
        path = task_cfg.ann
        self.coco = COCO(path)
        self.image_set_index = []
        for img in self.coco.dataset["images"]:
            self.image_set_index.append(img["id"])

        # === Db cfgs === #
        self.db = self._load_coco_keypoint_annotations(task_cfg)

        # === Augmentation cfgs === #
        augmentations = task_cfg.get(
            "augmentation", {"ImageNormalization": {"type": "base"}}
        )
        self.transform = ImageAugmentation(augmentations, mode="keypoint")

        self.num_images = len(self.image_set_index)
        logger.info(f"=> num_images: {self.num_images}")

        self.flip_pairs = self._get_flip_pair()

        logger.info(f"=> load {len(self.db)} samples")

    def _load_coco_keypoint_annotations(self, task_cfg):
        """ground truth bbox and keypoints"""
        gt_db = []
        for index in self.image_set_index:
            gt_db.extend(
                self._load_coco_keypoint_annotation_kernal(index, task_cfg)
            )
        return gt_db

    def _load_coco_keypoint_annotation_kernal(self, index, task_cfg):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: db entry
        """
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann["width"]
        height = im_ann["height"]

        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        # === Sanitize bboxes === #
        # NOTE: The sanitized bbox is used only for the center and scale
        #   calculation
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj["bbox"]
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1,
                ]  # Prevent the bbox over the img.
                valid_objs.append(obj)
        objs = valid_objs

        rec = []
        for obj in objs:

            # ignore objs without keypoints annotation
            if obj["num_keypoints"] == 0:
                continue

            # === Joint locations and their visibilities === #
            joints_3d = np.zeros((self.num_joints, 3))
            joints_3d_vis = np.zeros((self.num_joints, 3))
            for ipt in range(self.num_joints):
                joints_3d[ipt, 0] = obj["keypoints"][ipt * 3 + 0]
                joints_3d[ipt, 1] = obj["keypoints"][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0

                t_vis = obj["keypoints"][
                    ipt * 3 + 2
                ]  # TODO 나중에 정책 정해야 할듯.. ??
                if t_vis > 1:
                    t_vis = 1
                joints_3d_vis[ipt, 0] = t_vis
                joints_3d_vis[ipt, 1] = t_vis
                joints_3d_vis[ipt, 2] = 0

            # center, scale = self._box2cs(obj["clean_bbox"][:4])
            raw_box = np.array(obj["bbox"]).astype(float)
            file_path = os.path.join(task_cfg.data_root, im_ann["file_name"])
            rec.append(
                {
                    "image": file_path,
                    "center": -1,
                    "scale": -1,
                    "joints_3d": joints_3d,
                    "joints_3d_vis": joints_3d_vis,
                    "raw_box": raw_box,
                }
            )

        return rec

    def _get_img(
        self, raw_img, bbox, keypoints
    ) -> Tuple[np.ndarray, tuple, tuple, np.ndarray]:
        x_min, y_min, bbox_width, bbox_height = bbox
        x_max, y_max = x_min + bbox_width, y_min + bbox_height
        crop_img = raw_img[int(y_min) : int(y_max), int(x_min) : int(x_max), :]

        for joint_id in range(len(keypoints)):
            kp_x, kp_y, _ = keypoints[joint_id]
            is_visible = np.any([kp_x, kp_y])
            if is_visible:
                keypoints[joint_id] = [kp_x - x_min, kp_y - y_min, 0]

        if self.letter_box:
            r = (
                self.img_size[0] / bbox_width
                if bbox_width > bbox_height
                else self.img_size[1] / height
            )
            crop_img = cv2.resize(
                crop_img, (int(bbox_width * r), int(bbox_height * r))
            )
            ratio = (r, r)
            img, _, pad = letterbox(crop_img, self.img_size, auto=False)
            for joint_id in range(len(keypoints)):
                keypoints[joint_id] *= r
                keypoints[joint_id] += [pad[0], pad[1], 0]

        else:
            img = cv2.resize(crop_img, self.img_size)

            ratio = (
                self.img_size[0] / bbox_width,
                self.img_size[1] / bbox_height,
            )
            pad = (0, 0)
            for joint_id in range(len(keypoints)):
                keypoints[joint_id] *= [ratio[0], ratio[1], 0]

        keypoints = np.rint(keypoints)

        return img, ratio, pad, keypoints

    def __len__(self):
        # return len(self.db)  # num of persons
        return self.num_images  # num of images

    def __getitem__(self, idx) -> Tuple[torch.Tensor, dict]:
        db_rec = copy.deepcopy(self.db[idx])

        # === Unpack the db record === #
        image_file = db_rec["image"]
        joints_raw = db_rec["joints_3d"]
        box_raw = db_rec["raw_box"]  # Original bbox.

        # === Load the img === #
        img_raw = cv2.imread(image_file)
        if img_raw is None:
            logger.error(f"=> fail to read {image_file}")
            raise ValueError(f"Fail to read {image_file}")
        img, ratio, pad, joints = self._get_img(
            img_raw,
            box_raw,
            joints_raw,
        )  # Cropped img and keypoints.
        img_aug, joints_aug = self.transform.transform(
            img=img,
            labels=joints,
        )  # Augmented img and keypoints.
        img_aug = img2tensor(img_aug)
        joints_aug = self.check_flip_fair(
            joints_aug,
            joints,
        )  # Re-matching the left/right if flipped.

        # === Get the heatmap GT === #
        """
        # target (array): Gaussian heatmap intensities for joints
        # target_weight (bool): Presence of the joint to learn
        """
        target_heatmap, target_weight = self.generate_target(joints_aug)

        target_heatmap = torch.from_numpy(target_heatmap)
        # target_weight = torch.from_numpy(target_weight)

        outs = {
            "image": image_file,
            "img_size": self.img_size,
            "heatmap": target_heatmap,
            "joints": joints_aug,
            "ratio": ratio,
            "pad": pad,
            "raw_box": box_raw,
        }

        return img_aug, outs

    def generate_target(self, joints):
        """
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        Target type is gaussian heatmap.
        """
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)

        target = np.zeros(
            (self.num_joints, self.heatmap_size[1], self.heatmap_size[0]),
            dtype=np.float32,
        )

        tmp_size = self.sigma * 3

        for joint_id in range(self.num_joints):
            feat_stride = [
                self.img_size[0] / self.heatmap_size[0],
                self.img_size[1] / self.heatmap_size[1],
            ]
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if (
                ul[0] >= self.heatmap_size[0]
                or ul[1] >= self.heatmap_size[1]
                or br[0] < 0
                or br[1] < 0
            ):
                # If not, just return the image as is
                target_weight[joint_id, 0] = 0
                continue

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            v = target_weight[joint_id, 0]
            if v > 0:
                target[joint_id][
                    img_y[0] : img_y[1], img_x[0] : img_x[1]
                ] = self.g[g_y[0] : g_y[1], g_x[0] : g_x[1]]

        return target, target_weight

    def _get_joint_dict(self):
        joint_dict = {}
        for i, cls in enumerate(self.classes):
            joint_dict.update({i: cls})
        return joint_dict

    def _get_flip_pair(self) -> list:
        """It returns joint pairs of the right and left.

        Right comes first.

        Raises:
            ValueError: One of the joint pairs in ``self.class`` has either
                only left or right.

        Returns:
            flip_pairs (list): Pairs of the right and left joint idx.

        Example:
            flip_pairs = [
                [2, 1],  # right_eye, left_eye
                [4, 3],  # right_ear, left ear
                ... ,
                [16, 15],  # right_ankle, left_ankle
                ]
        """
        flip_pairs_dict = {}
        for i, j in enumerate(self.classes):
            if ("left" in j) or ("right" in j):
                position, joints_name = j.split("_")

                if joints_name not in flip_pairs_dict:
                    flip_pairs_dict.update(
                        {joints_name: [-1, -1]}
                    )  # Init the pair with -1

                if position == "left":
                    flip_pairs_dict[joints_name][1] = i
                elif position == "right":
                    flip_pairs_dict[joints_name][0] = i

        flip_pairs = []
        for k, v in flip_pairs_dict.items():
            if -1 in v:
                raise ValueError(
                    f"keypoint {k} seems to be left-right pair but only one \
                        keypoint exists"
                )
            flip_pairs.append(v)

        return flip_pairs

    def check_flip_fair(self, joints, joints_ori):
        new_joints = copy.deepcopy(joints)
        for pair in self.flip_pairs:
            r_idx, l_idx = pair
            cur_r, cur_l = joints[r_idx], joints[l_idx]
            ori_r, ori_l = joints_ori[r_idx], joints_ori[l_idx]
            if (cur_r[0] - cur_l[0]) * (ori_r[0] - ori_l[0]) < 0:
                new_joints[r_idx] = cur_l
                new_joints[l_idx] = cur_r
        return new_joints
