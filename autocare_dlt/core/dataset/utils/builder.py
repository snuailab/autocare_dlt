import sys

from autocare_dlt.core.dataset import *


def build_dataset(data_cfg, dataset_cfg):
    dataset_type = dataset_cfg.type
    dataset = getattr(sys.modules[__name__], dataset_type)(
        data_cfg, dataset_cfg
    )
    return dataset


def build_datasets(data_cfg):
    dataset_dict = {}
    if data_cfg.get("train", False):
        dataset_dict.update({"train": build_dataset(data_cfg, data_cfg.train)})
    if data_cfg.get("val", False):
        dataset_dict.update({"val": build_dataset(data_cfg, data_cfg.val)})
    if data_cfg.get("test", False):
        dataset_dict.update({"test": build_dataset(data_cfg, data_cfg.test)})

    return dataset_dict
