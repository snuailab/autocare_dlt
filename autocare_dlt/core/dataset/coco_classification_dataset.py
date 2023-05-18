import os
from collections import OrderedDict
from typing import List

import torch
from loguru import logger

from autocare_dlt.core.dataset.coco_base_dataset import COCOBaseDataset
from autocare_dlt.core.dataset.utils.functions import img2tensor, read_img


class COCOClassificationDataset(COCOBaseDataset):
    """COCO Base Dataset class"""

    def __init__(self, cfg: dict, task_cfg: dict):
        """init

        Args:
            cfg (dict): cfg
            task_cfg (dict): _description_
        """
        super().__init__(cfg, task_cfg)

    def load_annotations(self):
        """Load Annotations from coco format.
        Format definition is in
        https://www.notion.so/snuailab/Data-Format-Convention-COCO-Is-All-You-Need-7547fda8c1ca48798d00bd4658ea96bf.
        """

        # class <-> id mapper
        self.class_id = OrderedDict()
        self.id_class = OrderedDict()
        self.count_dict = OrderedDict()
        for attribute, labels in self.classes[0].items():
            self.class_id[attribute] = OrderedDict()
            self.id_class[attribute] = OrderedDict()
            self.count_dict[attribute] = OrderedDict()
            for i, label in enumerate(labels, start=0):
                self.class_id[attribute][label] = i
                self.id_class[attribute][i] = label
                self.count_dict[attribute][label] = 0

        # parse dataset
        self.datas = []
        for img_id, img_info in self.coco.imgs.items():
            data = {}
            data["image_path"] = os.path.join(
                self.data_root, img_info["file_name"]
            )

            ann_infos = self.coco.imgToAnns[img_id]
            if len(ann_infos) != len(self.classes):
                continue

            data["categories"] = []
            for target_attribute in self.classes[0].keys():
                for ann_info in ann_infos:
                    cat_id = ann_info["category_id"]
                    cat_info = self.coco.cats[cat_id]

                    attribute = cat_info["supercategory"]
                    if attribute != target_attribute:
                        continue

                    label = cat_info["name"]
                    if not (
                        attribute in self.class_id
                        and label in self.class_id[attribute]
                    ):
                        continue

                    data["categories"].append(self.class_id[attribute][label])
                    self.count_dict[attribute][label] += 1
                    break

            if len(data["categories"]) == len(self.classes):
                self.datas.append(data)
            else:
                logger.warning(
                    f'Some required label missing {data["image_path"]}'
                )

        # print counts
        for property, counts in self.count_dict.items():
            print(f"Property {property}")
            for option, count in counts.items():
                print(f"{option:>12} -> {count:<6}")

    def __len__(self) -> int:
        """available data num

        Returns:
            int: dataset length
        """
        return len(self.datas)

    def __getitem__(self, i: int) -> List[torch.Tensor]:
        """get item

        Args:
            i (int): index

        Returns:
            List[torch.Tensor]: input, target
        """

        data = self.datas[i]

        img, _, _ = read_img(data["image_path"], self.img_size)
        img, _ = self.transform.transform(img)
        img = img2tensor(img)

        label = torch.LongTensor(data["categories"])

        return img, label
