import os
from typing import List

import torch
from loguru import logger

from autocare_dlt.core.dataset.coco_base_dataset import COCOBaseDataset
from autocare_dlt.core.dataset.utils.functions import img2tensor, read_img


class COCOTextRecognitionDataset(COCOBaseDataset):
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

        max_string_length = self.cfg.get("max_string_length", 10)
        self.lpr = self.cfg.get("mode", "none") == "lpr"

        char2idx = {}
        char2idx["__BLANK__"] = 0
        for idx, char in enumerate(
            self.cfg.classes, start=1
        ):  # idx 0 is blank
            char2idx[char] = idx

        # parse dataset
        self.datas = []
        for img_id, img_info in self.coco.imgs.items():
            data = {}
            data["image_path"] = os.path.join(
                self.data_root, img_info["file_name"]
            )

            ann_infos = self.coco.imgToAnns[img_id]

            data["label"] = []
            for caption_char in ann_infos[0]["caption"]:
                data["label"].append(char2idx[caption_char])

            if self.lpr:
                if len(data["label"]) == 9:
                    data["label"].insert(2, 0)
                else:
                    while len(data["label"]) < max_string_length:
                        data["label"].insert(0, 0)
            else:
                while len(data["label"]) < max_string_length:
                    data["label"].append(0)

            if len(ann_infos):
                self.datas.append(data)
            else:
                logger.warning(
                    f'Some required label missing {data["image_path"]}'
                )

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
        img = img2tensor(img).float()

        if self.lpr:
            label = torch.LongTensor(data["label"])
        else:
            label = torch.IntTensor(data["label"])

        return img, label
