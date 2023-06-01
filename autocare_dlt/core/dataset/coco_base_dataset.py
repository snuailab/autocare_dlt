from pycocotools.coco import COCO

from autocare_dlt.core.dataset.utils.transforms import ImageAugmentation
from autocare_dlt.utils.config import (
    classifier_list,
    detector_list,
    regressor_list,
    str_list,
    segmenter_list
)


class COCOBaseDataset:
    """COCO Base Dataset class"""

    def __init__(self, cfg: dict, task_cfg: dict):
        """init

        Args:
            cfg (dict): cfg
            task_cfg (dict): _description_
        """

        self.cfg = cfg
        self.task_cfg = task_cfg

        # task
        if self.cfg.task in detector_list:
            self.mode = "detection"
        elif self.cfg.task in classifier_list:
            self.mode = "classification"
        elif self.cfg.task in regressor_list:
            self.mode = "regression"
        elif self.cfg.task in str_list:
            self.mode = "text_recognition"
        elif self.cfg.task in segmenter_list:
            self.mode = "segmentation"
            print(self.cfg.task)
            print(self.mode)
        else:
            raise BaseException(
                f"COCODataset does not support {self.cfg.task}"
            )

        # classes
        self.classes = cfg.get("classes", None)

        # image size
        self.img_size = self.cfg.get("img_size", [224, 224])
        if (not isinstance(self.img_size, list)) and (
            not isinstance(self.img_size, tuple)
        ):
            raise ValueError(
                f"data.img_size: {self.img_size} shuold be list of integer"
            )
        if len(self.img_size) == 1:
            self.img_size = [self.img_size[0], self.img_size[0]]

        # coco
        self.data_root = self.task_cfg.get("data_root", "")
        self.coco = COCO(self.task_cfg.ann)

        # transform
        augmentations = self.task_cfg.get(
            "augmentation", {"ImageNormalization": {"type": "base"}}
        )
        self.transform = ImageAugmentation(augmentations, mode=self.mode)

        # load annotations
        self.load_annotations()

    def load_annotations(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass
