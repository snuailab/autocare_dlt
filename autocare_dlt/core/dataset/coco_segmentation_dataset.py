import copy
import os

import numpy as np
import torch
from pycocotools.coco import COCO
import pycocotools.mask as mask_utils
from torch.utils.data import Dataset
from tqdm import tqdm
import cv2

from autocare_dlt.core.dataset.utils import *


class COCOSegmentationDataset(Dataset):
    def __init__(self, cfg, task_cfg):
        # COCO loader
        self.img_size = cfg.img_size

        if (not isinstance(self.img_size, list)) and (
            not isinstance(self.img_size, tuple)
        ):
            raise ValueError(
                f"data.img_size: {self.img_size} shuold be list of integer"
            )
        self.task_cfg = task_cfg
        self.data_root = self.task_cfg.data_root

        self.img_size = (
            [self.img_size[0], self.img_size[0]]
            if len(self.img_size) == 1
            else self.img_size
        )
   
        path = self.task_cfg.ann
        self.coco = COCO(path)

        self.classes = cfg.get("classes")

        # Get labels
        labels, shapes = self.load_annotations(self.classes)

        self.shapes = np.array(shapes, dtype=np.float64)
        self.labels = list(labels)
        
        self.data_statistics(path)

        augmentations = self.task_cfg.get(
            "augmentation", {"ImageNormalization": {"type": "base"}}
        )

        pad = augmentations.get("Pad", False)
        if pad:
            self.pad = pad.get("ratio", 0.0)

        self.transform = ImageAugmentation(augmentations, mode="segmentation")

    def data_statistics(self, path):
        n = len(self.img_files)
        print(f"Scanning labels {path} (labels for {n} images)")

    def load_annotations(self, classes):
        labels = []
        shapes = []
        self.img_files = []
        self.available_ids = []
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self.data_classes = tuple([c["name"] for c in self.cats])
        self.data_class_ids = tuple([c["id"] for c in self.cats])
        self.cls_mapping = [None] * len(classes)
        img_ids = self.coco.getImgIds()        
        for i, c in enumerate(classes):
            self.cls_mapping[i] = self.data_class_ids[
                self.data_classes.index(c)
            ]

        for id_ in img_ids:
            im_ann = self.coco.loadImgs(id_)[0]
            width = im_ann["width"]
            height = im_ann["height"]
            anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
            annotations = self.coco.loadAnns(anno_ids)

            seg_label = np.full((height, width), len(classes))
            for obj in annotations:
                cls_index = self.data_class_ids.index(obj["category_id"])
                cls = self.data_classes[cls_index]
                if cls not in classes:
                    continue

                mask = self.coco.annToMask(obj)
                loc = mask == 1
                seg_label[loc] = cls_index

            if len(np.unique(seg_label)) == 1:
                continue

            l = np.array(seg_label, dtype=np.float32)
            labels.append(l)
            shapes.append(np.array([width, height]))
            self.img_files.append(self.coco.loadImgs(id_)[0]["file_name"])
            self.available_ids.append(id_)

        n = len(self.img_files)
        print(f"number of img_files in DB: {len(self.img_files)}")

        self.n = n  # number of images
        return labels, shapes

    def load_image(self, index):
        path = os.path.join(self.data_root, self.img_files[index])
        img, h0, w0 = (
            read_img(path, self.img_size)
        )
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized

    def segmask_resize(self, label):
        label = cv2.resize(label, self.img_size, interpolation=cv2.INTER_NEAREST)
        return label
    
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):

        # Load image
        img, (h0, w0), (h, w) = self.load_image(index)

        # Load & transform labels
        labels = self.labels[index]
        labels = self.segmask_resize(labels)

        # Padding
        pad = (int(h*self.pad), int(h*self.pad), int(w*self.pad), int(w*self.pad))

        img = np.pad(img, ((int(h*self.pad), int(h*self.pad)), (int(w*self.pad), int(w*self.pad)), (0, 0)), "constant", constant_values=0)
        labels = np.pad(labels, ((int(h*self.pad), int(h*self.pad)), (int(w*self.pad), int(w*self.pad))), "constant", constant_values=len(self.classes))

        # Converting Background Channels for Use with Albumentation
        labels = np.where(labels == len(self.classes), 0, labels+1)
        img, labels = self.transform.transform(img, labels)
        labels = torch.where(labels == 0, torch.tensor(len(self.classes)), labels - 1)

        img = img2tensor(img)
   
        outs = {
            "labels": labels,  # (n, 1)
            "ori_shape": (h0, w0),
            "ratio": ((h / h0, w / w0), pad),
            "path": self.img_files[index],
        }

        return img, outs
