import copy
import os

import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from tqdm import tqdm

from autocare_dlt.core.dataset.utils import *


class COCODetectionDataset(Dataset):
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
        self.letter_box = cfg.get("letter_box", False)
        if self.letter_box:
            if len(self.img_size) > 1:
                print(
                    f"WARNING: rectification is activated, img size is set to [{self.img_size[0]}, {self.img_size[0]}]"
                )
            self.img_size = self.img_size[0]
        else:
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

        self.transform = ImageAugmentation(augmentations)

    def data_statistics(self, path):
        n = len(self.img_files)
        nm, nf, ne, nd = (
            0,
            0,
            0,
            0,
        )  # number missing, found, empty, datasubset, duplicate
        pbar = tqdm(self.labels)
        for i, l in enumerate(pbar):
            if l.shape[0]:
                assert l.shape[1] == 5, (
                    "> 5 label columns: %s" % self.img_files[i]
                )
                assert (l >= 0).all(), (
                    "negative labels: %s" % self.img_files[i]
                )
                # assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % self.img_files[i]
                if (
                    np.unique(l, axis=0).shape[0] < l.shape[0]
                ):  # duplicate rows
                    nd += 1  # duplicate rows
                nf += 1  # file found
            else:
                ne += 1  # print('empty labels for image %s' % self.img_files[i])  # file empty

            pbar.desc = "Scanning labels {} ({:g} found, {:g} missing, {:g} empty, {:g} duplicate, for {:g} images)".format(
                path, nf, nm, ne, nd, n
            )

        if nf == 0:
            s = "WARNING: No labels found in %s" % (
                os.path.dirname(self.img_files[0]) + os.sep
            )
            print(s)

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
            objs = []

            for obj in annotations:
                cls_index = self.data_class_ids.index(obj["category_id"])
                cls = self.data_classes[cls_index]
                if cls not in classes:
                    continue
                x1 = np.max((0, obj["bbox"][0]))
                y1 = np.max((0, obj["bbox"][1]))
                x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
                y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
                if x2 > x1 and y2 > y1:
                    # obj["clean_bbox"] = [x1, y1, x2, y2]
                    obj_width = x2 - x1
                    obj_height = y2 - y1
                    x_center = x1 + 0.5 * obj_width
                    y_center = y1 + 0.5 * obj_height
                    cls_label = classes.index(cls)
                    objs.append(
                        [
                            cls_label,
                            x_center / width,
                            y_center / height,
                            obj_width / width,
                            obj_height / height,
                        ]
                    )
            if len(objs) == 0:
                continue
            l = np.array(objs, dtype=np.float32)
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
            read_img_rect(path, self.img_size)
            if self.letter_box
            else read_img(path, self.img_size)
        )
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized

    def norm_xywh2xyxy(self, labels, ratio, pad, shape):
        labels = labels.copy()
        (h, w) = shape
        if len(labels) > 0:
            # Normalized xywh to pixel xyxy format
            x = labels.copy()
            labels[:, 1] = (
                ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]
            )  # pad width
            labels[:, 2] = (
                ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]
            )  # pad height
            labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
            labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]
        return labels

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):

        # Load image
        img, (h0, w0), (h, w) = self.load_image(index)
        if self.letter_box:
            img, ratio, pad = letterbox(
                img, (self.img_size, self.img_size), auto=False
            )
        else:
            ratio = (1, 1)
            pad = (0, 0)

        # Load & transform labels
        labels = self.norm_xywh2xyxy(self.labels[index], ratio, pad, (h, w))
        img, labels = self.transform.transform(img, labels)

        img = img2tensor(img)

        num_labels = len(labels)

        if num_labels > 0:
            bbox = labels[:, 1:5]
            labels = labels[:, 0].type(torch.long)
        else:
            bbox = None
            labels = None

        outs = {
            "labels": labels,  # (n, 1)
            "boxes": bbox,  # (n, 4)
            "ori_shape": (h0, w0),
            "ratio": ((h / h0, w / w0), pad),
            "path": self.img_files[index],
        }

        return img, outs
