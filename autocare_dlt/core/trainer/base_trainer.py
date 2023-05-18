import math
import os

import torch
import torch.nn as nn
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP

from autocare_dlt.core.loss.loss_manager import (
    ClsLossManager,
    DetLossManager,
    PoseLossManager,
    STRLossManager,
)
from autocare_dlt.core.model.classifier import *
from autocare_dlt.core.model.detector import *
from autocare_dlt.core.model.pose_estimation import *
from autocare_dlt.core.model.regressor import *
from autocare_dlt.core.model.text_recognition import *
from autocare_dlt.core.utils import (
    AverageMeter,
    Inferece,
    LRScheduler,
    load_ckpt,
    save_checkpoint,
)
from autocare_dlt.utils.config import (
    classifier_list,
    detector_list,
    pose_estimator_list,
    regressor_list,
    save_cfg,
    str_list,
)


class BaseTrainer:
    def __init__(self, model, datasets, cfg):

        self.model = model
        self.mode = None
        self.datasets = datasets

        if not cfg.get("optim", False):
            raise ValueError("Can not found optimizer config(=cfg.optim).")
        self.ema = cfg.ema

        # === Trainer Common Configs ===#
        self.start_epoch = cfg.get("start_epoch", 0)
        self.max_epoch = cfg.get("max_epoch", 12)
        self.max_iter = cfg.get("max_iter", 0)
        self.eval_interval = cfg.get("eval_interval", 1)

        self.train_unit = "iter" if self.max_iter else "epoch"

        self.data_type = torch.float32

        # === Distributed Learning ===#
        # TODO https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#basic-use-case
        self.num_gpus = cfg.num_gpus

        self.distributed = True if len(cfg.gpus.split(",")) > 1 else False
        if self.distributed:
            self.gpus = cfg.gpus
            self.gpu_id = cfg.get("gpu_id", int(cfg.gpus[0]))
            self.world_size = cfg.world_size
        else:
            self.gpu_id = -1
        self.rank = cfg.get("rank", 0)

        # === early stop ===#
        self.early_stop_cnt_max = cfg.get("early_stop_count", False)
        self.early_stop_cnt_stack = 0

        # === Inferencer ===#
        self.inference = Inferece(cfg)

        self.verbose = cfg.get("verbose", False)

        self.cfg = cfg
        self.output_path = self.cfg.output_path
        if self.rank == 0:
            save_cfg(self.output_path, self.cfg)

        logger.add(os.path.join(self.output_path, f"{cfg.exp_name}.log"))

    def _get_dataloader(self):
        pass

    def cuda(self):
        self.model.cuda() if not (self.distributed) else self.model.cuda(
            self.rank
        )
        if self.cfg.ema:
            self.ema_model.ema.cuda() if not (
                self.distributed
            ) else self.ema_model.ema.cuda(
                self.rank
            )  # not so sure
        self.loss_manager.cuda() if not (
            self.distributed
        ) else self.loss_manager.cuda(-1)

    def _get_optimizer(self):
        self.start_lr = self.cfg.optim.get("lr", 0.01)
        weight_decay = self.cfg.optim.get("weight_decay", 0)

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in self.model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay

        name = self.cfg.optim.get("name", None)
        if name == "SGD":
            momentum = (
                0.9
                if self.cfg.optim.get("momentum", None) is None
                else self.cfg.optim.momentum
            )
            self.optimizer = torch.optim.SGD(
                pg0, lr=self.start_lr, momentum=momentum, nesterov=True
            )
        elif name == "Adam":
            self.optimizer = torch.optim.Adam(pg0, lr=self.start_lr)
        elif name == "Adadelta":
            rho = (
                0.95
                if self.cfg.optim.get("rho", None) is None
                else self.cfg.optim.rho
            )
            eps = (
                1e-8
                if self.cfg.optim.get("eps", None) is None
                else self.cfg.optim.eps
            )
            self.optimizer = torch.optim.Adadelta(
                pg0, lr=self.start_lr, rho=rho, eps=eps
            )
        else:
            raise ValueError(f"optimizer: {name} is not supported")

        self.optimizer.add_param_group(
            {"params": pg1, "weight_decay": weight_decay}
        )  # add pg1 with weight_decay
        self.optimizer.add_param_group({"params": pg2})  # add pg2 (biases)
        if self.rank == 0:
            logger.info(
                "Optimizer groups: %g .bias, %g conv.weight, %g other"
                % (len(pg2), len(pg1), len(pg0))
            )
        del pg0, pg1, pg2

    def _get_lr_scheduler(self, start_lr, iters_per_epoch):
        self.lr = start_lr
        if self.cfg.get("lr_cfg", False):
            self.lr_scheduler = LRScheduler(
                self.cfg.lr_cfg.get("type", None),
                start_lr,
                iters_per_epoch,
                self.max_epoch,
                warmup=self.cfg.lr_cfg.get("warmup", False),
                warmup_epochs=self.cfg.lr_cfg.get("warmup_epochs", 1),
                warmup_lr_start=self.cfg.lr_cfg.get("warmup_lr", 1e-6),
                steps=self.cfg.lr_cfg.get("steps", None),
                decay=self.cfg.lr_cfg.get("decay", 0.1),
            )
        else:
            self.lr_scheduler = None

    def _get_loss_fn(self):
        loss_cfg = self.cfg.get("loss", None)

        if loss_cfg is None:
            raise BaseException("Loss config missing")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.cfg.get("task") in classifier_list:
            self.loss_manager = ClsLossManager(loss_cfg, device=device)
        elif (
            self.cfg.get("task") in regressor_list
        ):  # TODO: merge to classification
            self.loss_manager = ClsLossManager(loss_cfg, device=device)
        elif self.cfg.get("task") in detector_list:
            self.loss_manager = DetLossManager(loss_cfg)
        elif self.cfg.get("task") in str_list:
            self.loss_manager = STRLossManager(loss_cfg)
        elif self.cfg.get("task") in pose_estimator_list:
            self.loss_manager = PoseLossManager(loss_cfg, device=device)

    def update_lr(self, iter):
        if self.lr_scheduler is not None:
            self.lr = self.lr_scheduler.update_lr(iter)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr

    def train(self):
        if self.rank == 0:
            logger.info("Training start...")
        self.before_train()
        self.run_train()
        self.after_train()
        return self.model

    def run_train(self):
        if self.train_unit == "iter":
            self.max_epoch = math.ceil(self.max_iter / self.iters_per_epoch)

        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.loss_aver = AverageMeter()
            self.acc_aver = AverageMeter()

            self.before_epoch()
            self.run_epoch()
            self.after_epoch()

            # early stop
            if self.early_stop_cnt_max != False:
                if self.early_stop_cnt_stack >= self.early_stop_cnt_max:
                    logger.info(f"early stopped (epoch : {self.epoch})")
                    break

    def run_epoch(self):
        for self.iter in range(1, self.iters_per_epoch + 1):
            self.before_iter()
            self.run_iter()
            self.after_iter()

    def run_iter(self):
        pass

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def before_iter(self):
        pass

    def after_iter(self):
        pass

    def progress_in_iter(self):
        return self.epoch * self.iters_per_epoch + self.iter

    def resume_train(self):
        if self.cfg.resume:

            if self.rank == 0:
                logger.info("resume training")

            ckpt_file = os.path.join(self.output_path, "last_epoch_ckpt.pth")
            assert os.path.isfile(ckpt_file)

            ckpt = torch.load(ckpt_file, map_location="cpu")

            # resume the model/optimizer state dict
            self.model = load_ckpt(self.model, ckpt.get("model", {}))
            if self.cfg.ema:
                self.ema_model.ema = load_ckpt(
                    self.ema_model.ema,
                    ckpt["model"]
                    if ckpt.get("model_ema", None) is None
                    else ckpt["model_ema"],
                )

            # resume the training states variables
            self.start_epoch = ckpt.get("start_epoch", self.start_epoch)
            if self.rank == 0:
                logger.info(
                    "loaded checkpoint '{}' (epoch {})".format(
                        self.cfg.resume, self.start_epoch
                    )
                )

    def evaluate_and_save_model(self):
        pass

    def test_model(self):
        pass

    def save_ckpt(
        self,
        save_model,
        model_ema=None,
        ckpt_name="last_epoch",
        update_best_ckpt=False,
    ):
        logger.info(f"Save weights to {self.output_path}")
        ckpt_state = {
            "start_epoch": self.epoch + 1,
            "model": save_model.state_dict()
            if not (hasattr(save_model, "module"))
            else save_model.module.state_dict(),  # save nonDDP model ckpt
            "model_ema": None if model_ema is None else model_ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        save_checkpoint(
            ckpt_state,
            update_best_ckpt,
            self.output_path,
            ckpt_name,
        )

        # stack early stop count
        if self.early_stop_cnt_max != False:
            if update_best_ckpt == True:
                self.early_stop_cnt_stack = 0
            else:
                self.early_stop_cnt_stack += 1
            logger.info(
                f"early stop count stacked : {self.early_stop_cnt_stack}/{self.early_stop_cnt_max}"
            )
