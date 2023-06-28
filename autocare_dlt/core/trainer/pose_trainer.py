import os
import sys

import numpy as np
import torch
from loguru import logger

from autocare_dlt.core.dataset.utils import (
    DataIterator,
    collate_fn,
)
from autocare_dlt.core.dataset.utils.pose_eval import pck_accuracy
from autocare_dlt.core.model.utils.functions import is_parallel
from autocare_dlt.core.trainer import BaseTrainer
from autocare_dlt.core.utils import key_labels_to_cuda
from autocare_dlt.core.utils.functions import check_gpu_availability


class PoseTrainer(BaseTrainer):
    def __init__(self, model, datasets, cfg):
        """_summary_

        TODO: Description.

        Args:
            cfg (_type_): Configs defined in ``utils/config.py``.
        """
        super().__init__(model, datasets, cfg)

        # === Trainer cfgs === #
        self.best_score = 0
        self.use_model_ema = cfg.get("use_model_ema", False)
        self.eval_interval = cfg.get("eval_interval", 1)
        self.full_arch_name = (
            f"backbone: {cfg.get('model').get('backbone').get('name')} | "
            + f"neck: {cfg.get('model').get('neck').get('name')} | "
            + f"head: {cfg.get('model').get('head').get('name')}"
        )

        # === Monitor cfgs === #
        # NOTE: hardcoded now because we don't have a cfg json for vis
        self.log_freq = 500

        # === Init values === #
        self.best_ap = -1

    def _get_dataloader(self):
        data_cfg = self.cfg.data

        if self.datasets.get("train", False):
            self.train_dataset = self.datasets["train"]
            self.train_dataloader = torch.utils.data.DataLoader(  # type: ignore
                self.train_dataset,
                batch_size=data_cfg.batch_size_per_gpu,
                num_workers=data_cfg.workers_per_gpu * self.num_gpus,
                shuffle=True,
                pin_memory=True,
                collate_fn=collate_fn,
            )

        if self.datasets.get("val", False):
            self.val_dataset = self.datasets["val"]
            self.val_dataloader = torch.utils.data.DataLoader(  # type: ignore
                self.val_dataset,
                batch_size=data_cfg.batch_size_per_gpu,
                num_workers=data_cfg.workers_per_gpu,
                shuffle=False,
                pin_memory=True,
                collate_fn=collate_fn,
            )

        if self.datasets.get("test", False):
            self.test_dataset = self.datasets["test"]
            self.test_dataloader = torch.utils.data.DataLoader(  # type: ignore
                self.test_dataset,
                batch_size=data_cfg.batch_size_per_gpu,
                num_workers=data_cfg.workers_per_gpu,
                shuffle=False,
                pin_memory=True,
                collate_fn=collate_fn,
            )

    def before_train(self):

        self._get_dataloader()
        self.labeled_iter = DataIterator(self.train_dataloader)
        self.iters_per_epoch = self.cfg.get(
            "iters_per_epoch", len(self.train_dataloader)
        )

        # TODO: temporal functions, not pretty
        if self.cfg.ema:
            from autocare_dlt.core.model.utils.ema import ModelEMA

            self.burn_in_iters = (
                len(self.train_dataloader) * self.cfg.ema_cfg.burn_in_epoch
            )
            self.ema_model = ModelEMA(
                self.model,
                decay=self.cfg.ema_cfg.decay,
                max_iter=self.burn_in_iters,
            )
        if self.distributed:
            self.model = self.model.to(self.rank)

        self._get_optimizer()
        self._get_loss_fn()

        self.resume_train()

        self._get_lr_scheduler(self.start_lr, self.iters_per_epoch)

        # === GPU memory availability check === #
        if self.rank == 0 and torch.cuda.is_available():
            gpu_availability = check_gpu_availability(
                model=self.model,
                input_size=self.cfg.data.img_size,
                batch_size=self.cfg.data.batch_size_per_gpu,
                dtype=self.data_type,
                gpu_total_mem=torch.cuda.get_device_properties(
                    0
                ).total_memory,  # Bytes
            )  # Bool. # TODO: Consider allocated memory

            if not gpu_availability:
                sys.exit(-1)

        if torch.cuda.is_available():
            self.cuda()

        # TODO https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#basic-use-case
        # self.model = DDP(self.model.cuda(), device_ids=[torch.cuda.current_device()], broadcast_buffers=False)

    def after_train(self):
        # TODO: use appropriate eval metric
        logger.info(
            "Training of experiment is done and the best score is {:.2f}".format(
                self.best_score
            )
        )
        self.test_model()

    def before_epoch(self):
        logger.info(f"---> start train epoch{self.epoch}")
        self.model.train()

    def after_epoch(self):
        # loggers
        tags = ["train/lr", "train/loss", "train/accuracy"]
        values = [self.lr, self.loss_aver.avg, self.acc_aver.avg]
        for tag, value in zip(tags, values):
            self.tblogger.add_scalar(tag, value, self.epoch)

        self.evaluate_and_save_model()

    def test_model(self):
        # === Evaluate ===#
        logger.info("Test start...")

        self.cfg.resume = False
        self.cfg.ckpt = os.path.join(self.output_path, "best_ckpt.pth")
        self.resume_train()

        evalmodel = self.ema_model.ema if self.ema else self.model
        if is_parallel(evalmodel):
            evalmodel = evalmodel.module

        evalmodel.eval()

        res, labels = self.inference(evalmodel, self.test_dataloader)
        logger.info("Evaluate..")
        # TODO Pose eval function - Use PCK or COCO mAP?
        res = torch.cat(res).numpy()
        labels = np.concatenate(labels).tolist()
        # PCK
        _, score, cnt, pred = pck_accuracy(
            res,
            labels,
        )
        # COCO
        # res = convert_keypoints_to_coco(res, labels)
        # ap50_95, ap50, cls_ap_dict, summary = coco_evaluation(res_2point, self.test_dataloader.dataset, ann_type='keypoints', print_cls_ap=self.verbose)

        logger.info(f"Test PCK Score: {score}")
        tags = ["test/PCK"]
        values = [score]
        for tag, value in zip(tags, values):
            self.tblogger.add_scalar(tag, value, self.epoch)

    def evaluate_and_save_model(self):
        # === Evaluate ===#
        logger.info("Validation start...")

        evalmodel = self.ema_model.ema if self.ema else self.model
        if is_parallel(evalmodel):
            evalmodel = evalmodel.module

        evalmodel.eval()

        res, labels = self.inference(evalmodel, self.val_dataloader)
        logger.info("Evaluate..")
        # TODO Pose eval function - Use PCK or COCO mAP?
        # PCK
        res = torch.cat(res).numpy()
        labels = np.concatenate(labels).tolist()
        _, score, cnt, pred = pck_accuracy(
            res,
            labels,
        )
        # COCO
        # res = convert_keypoints_to_coco(res, labels)
        # ap50_95, ap50, cls_ap_dict, summary = coco_evaluation(res_2point, self.val_dataloader.dataset, ann_type='keypoints', print_cls_ap=self.verbose)

        logger.info(f"Validation PCK Score: {score}")

        tags = ["val/PCK"]
        values = [score]
        for tag, value in zip(tags, values):
            self.tblogger.add_scalar(tag, value, self.epoch)

        if score > self.best_score:
            self.best_score = score
            best_model = True
        else:
            best_model = False

        # === Save Checkpoints ===#
        self.save_ckpt(
            self.model,
            None if not self.ema else self.ema_model.ema,
            "last_epoch",
            best_model,
        )

    def before_iter(self):
        self.update_lr(self.progress_in_iter())

    def run_iter(self):
        loss = 0
        labeled_inputs, labeled_targets = self.labeled_iter()
        if torch.cuda.is_available():
            labeled_inputs = labeled_inputs.cuda()  # Input img
            key_labels_to_cuda(labeled_targets)

        sup_outputs = self.model(labeled_inputs)
        sup_loss, sup_loss_dict = self.loss_manager(
            sup_outputs, labeled_targets
        )
        loss += sup_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss_aver.update(loss)

        _, avg_acc, cnt, pred = pck_accuracy(
            sup_outputs.detach().cpu().numpy(), labeled_targets
        )
        self.acc_aver.update(avg_acc, cnt)

        if self.ema:
            self.ema_model.update(self.model, iter=self.progress_in_iter())

        # log
        if self.iter % 100 == 0 or self.iter == 1:
            train_mesg = f"epoch {self.epoch} [{self.iter}/{self.iters_per_epoch}] - lr: {self.lr:0.6f} train loss: {self.loss_aver.avg:0.6f}, train accuracy: {self.acc_aver.avg:0.3f} "
            logger.info(train_mesg)

        # visualize every first iteration
        if self.cfg.get("DEBUG_data_sanity", False) and self.iter == 1:
            pass
            # dirs = os.path.join(
            #         os.path.join("outputs", self.cfg.exp_name), "heatmap"
            #     )
            #     checkdir(dirs)
            #     prefix = "{}_{}".format(os.path.join(dirs, "train"), self.epoch)

            #     save_debug_images(
            #         inputs,
            #         targets,
            #         outputs,
            #         joint_labels,
            #         torch.Tensor(pred * 4),
            #         prefix,
            #     )

    def after_iter(self):
        pass
