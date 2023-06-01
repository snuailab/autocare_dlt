import argparse
import os
import random
import sys
import warnings
from typing import Union

import numpy as np
from box import Box

sys.path.append(os.getcwd())
import torch
import torch.backends.cudnn as cudnn
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from autocare_dlt.core.dataset import build_datasets
from autocare_dlt.core.model import build_model
from autocare_dlt.core.trainer import (
    ClassificationTrainer,
    DetectionTrainer,
    PoseTrainer,
    RegressionTrainer,
    StrTrainer,
    SegmentationTrainer
)
from autocare_dlt.core.utils import init_dist
from autocare_dlt.utils.config import (
    classifier_list,
    detector_list,
    parsing_config,
    pose_estimator_list,
    regressor_list,
    str2bool,
    str_list,
    segmenter_list
)


def make_parser():
    parser = argparse.ArgumentParser("Tx trainer")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/",
        help="log output directory",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="experiment name",
        required=True,
    )
    parser.add_argument(
        "--model_cfg",
        default=None,
        type=str,
        help="path for model configuation",
        required=True,
    )
    parser.add_argument(
        "--data_cfg",
        default=None,
        type=str,
        help="path for data configuation",
        required=True,
    )
    parser.add_argument(
        "--gpus", type=str, default="0", help="gpus for training"
    )
    parser.add_argument(
        "--ckpt", default=None, type=str, help="checkpoint file"
    )
    parser.add_argument(
        "--resume", default=False, type=str2bool, help="resume training"
    )
    parser.add_argument(
        "--ema",
        default=False,
        type=str2bool,
        help="use exponential moving average",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        type=str2bool,
        help="overwrite former results",
    )

    return parser


def procs(rank, model, datasets, cfg, return_dict=False):
    if cfg.seed is not None:
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        cudnn.deterministic = True
        np.random.seed(cfg.seed)
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )
        if cfg.gpus == "-1":
            torch.cuda.manual_seed(cfg.seed)
            cudnn.benchmark = True

    # set output_path
    output_path = os.path.join(cfg.output_dir, cfg.exp_name)
    cfg.output_path = output_path
    if (
        not cfg.get("overwrite", False)
        and not cfg.get("resume", False)
        and os.path.exists(output_path)
    ):
        raise KeyError(f"exp_name {cfg.exp_name} is already exist")

    # log path
    log_path = os.path.join(cfg.get("output_dir"), cfg.exp_name, "tensorboard")
    tblogger = SummaryWriter(log_path)

    if cfg.gpus != "-1" and len(cfg.gpus.split(",")) > 1:
        cfg.num_gpus = len(cfg.gpus.split(","))
        cfg.gpu_id = int(cfg.gpus[rank * 2])
        cfg.rank = rank
        init_dist(rank, cfg.world_size, cfg.gpu_id)
    else:
        cfg.num_gpus = 1  # single GPU or CPU mode

    if cfg.task in detector_list:  # Train w/ Tx
        trainer = DetectionTrainer(model, datasets, cfg)
    elif cfg.task in classifier_list:
        trainer = ClassificationTrainer(model, datasets, cfg)
    elif cfg.task in regressor_list:
        trainer = RegressionTrainer(model, datasets, cfg)
    elif cfg.task in str_list:
        trainer = StrTrainer(model, datasets, cfg)
    elif cfg.task in pose_estimator_list:
        trainer = PoseTrainer(model, datasets, cfg)
    elif cfg.task in segmenter_list:
        trainer = SegmentationTrainer(model, datasets, cfg)
    else:
        raise KeyError(f"cfg.task: {cfg.task} is unsupported task.")

    model = trainer.train()

    if return_dict != False and rank == 0:
        # TODO: return model (DDP model => model)
        return_dict["model"] = None

    else:
        return model


def distributed_procs(model, datasets, cfg, procs):
    if len(cfg.gpus.split(",")) == 1:
        if cfg.gpus != "-1" and torch.cuda.is_available():
            torch.cuda.set_device(device=f"cuda:{cfg.gpus}")
        else:
            torch.cuda.is_available = lambda: False
        model = procs(0, model, datasets, cfg)

    elif len(cfg.gpus.split(",")) > 1:
        if cfg.gpus == "-1":
            raise BaseException(
                "Distributed learning does not support cpu training."
            )
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus
        import torch.multiprocessing as mp

        logger.info("Multi GPU Training start...")
        cfg.world_size = int((len(cfg.gpus) + 1) / 2)
        manager = mp.Manager()
        return_dict = manager.dict()
        # TODO: quit mp.spawn for hpo with DDP
        mp.spawn(
            procs,
            args=(model, datasets, cfg, return_dict),
            nprocs=cfg.world_size,
            join=True,
        )

        model = return_dict["model"]

    return model


def run(
    exp_name: str,
    model_cfg: str,
    data_cfg: str,
    gpus: str = "0",
    ckpt: Union[str, dict] = None,
    world_size: int = 1,
    output_dir: str = "outputs",
    resume: bool = False,
    ema: bool = False,
    overwrite: bool = False,
) -> None:
    """Run training

    Args:

        exp_name (str): experiment name. a folder with this name will be created in the ``output_dir``, and the log files will be saved there.
        model_cfg (str): path for model configuration file
        data_cfg (str): path for dataset configuration file
        gpus (str, optional): GPU IDs to use. Default to '0'
        ckpt (str, optional): path for checkpoint file. Defaults to None.
        world_size (int, optional): world size for ddp. Defaults to 1.
        output_dir (str, optional): log output directory. Defaults to 'outputs'.
        resume (bool, optional): wheather to resume the previous training or not. Defaults to False.
        ema (bool, optional): wheather to use EMA(exponential moving average) or not. Defaults to False.
    """

    args = Box(
        {
            "output_dir": output_dir,
            "exp_name": exp_name,
            "model_cfg": model_cfg,
            "data_cfg": data_cfg,
            "gpus": gpus,
            "world_size": world_size,
            "ckpt": ckpt,
            "resume": resume,
            "ema": ema,
            "overwrite": overwrite,
        }
    )

    cfg = parsing_config(args)
    model, classes = build_model(cfg)
    datasets = build_datasets(cfg.data)

    model = distributed_procs(model, datasets, cfg, procs)

    return model


if __name__ == "__main__":
    args = make_parser().parse_args()
    run(**vars(args))
