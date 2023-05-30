import os
import sys

import torch
import torch.distributed as dist
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from autocare_dlt.core.dataset.utils import (
    DataIterator,
    coco_evaluation,
    collate_fn,
    convert_4pointBbox_to_coco_format,
    convert_to_coco_format,
)
from autocare_dlt.core.model.utils.functions import is_parallel
from autocare_dlt.core.trainer import BaseTrainer
from autocare_dlt.core.utils import (
    det_labels_to_cuda,
    load_ckpt,
    nms,
)
from autocare_dlt.core.utils.functions import check_gpu_availability
from autocare_dlt.utils.debugging import save_labels


class DetectionTrainer(BaseTrainer):
    def __init__(self, model, datasets, cfg):

        super().__init__(model, datasets, cfg)

        # === Trainer Configs ===#
        self.detections_per_img = self.cfg.get("detections_per_img", 100)
        self.nms_thresh = self.cfg.get("nms_thresh", 0.45)
        self.min_score = self.cfg.get("min_score", 0.01)
        self.target_cls = self.cfg.get("target_class", "dummy_class")

        # === Values ===#
        self.best_ap = -1

    def _get_dataloader(self):
        data_cfg = self.cfg.data
        if self.datasets.get("train", False):
            self.train_dataset = self.datasets["train"]
            self.train_sampler = (
                None
                if not (self.distributed)
                else DistributedSampler(self.train_dataset, shuffle=True)
            )
            self.train_dataloader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=data_cfg.batch_size_per_gpu,
                num_workers=data_cfg.workers_per_gpu * self.num_gpus,
                shuffle=True if not (self.distributed) else False,
                pin_memory=True,
                collate_fn=collate_fn,
                sampler=self.train_sampler,
            )
        if self.datasets.get("val", False):
            self.val_dataset = self.datasets["val"]
            self.val_dataloader = torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=data_cfg.batch_size_per_gpu,
                num_workers=data_cfg.workers_per_gpu,
                shuffle=False,
                pin_memory=True,
                collate_fn=collate_fn,
            )
        if self.datasets.get("test", False):
            self.test_dataset = self.datasets["test"]
            self.test_dataloader = torch.utils.data.DataLoader(
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
        if self.distributed:
            dist.barrier()
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
        if self.distributed:
            self.model = DDP(
                module=self.model,
                device_ids=[self.rank],
                output_device=self.rank,
                broadcast_buffers=True,
            )
        # TODO https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#basic-use-case
        # self.model = DDP(self.model.cuda(), device_ids=[torch.cuda.current_device()], broadcast_buffers=False)

    def after_train(self):
        if self.rank == 0:
            logger.info(
                f"Training of experiment is done and the best validation AP is {self.best_ap * 100:.2f}"
            )
            self.test_model()

    def before_epoch(self):
        if self.distributed:
            if self.rank == 0:
                logger.info(f"---> start train epoch{self.epoch}")

            self.train_sampler.set_epoch(self.epoch)
        else:
            logger.info(f"---> start train epoch{self.epoch}")
        self.model.train()

    def after_epoch(self):
        # loggers
        tags = ["train/lr", "train/loss"]
        values = [self.lr, self.loss_aver.avg]

        for tag, value in zip(tags, values):
            self.tblogger.add_scalar(tag, value, self.epoch)

        if not (self.distributed):
            self.evaluate_and_save_model()

        if self.distributed:
            if self.rank == 0:
                self.evaluate_and_save_model()
            # wait for master process to save model
            dist.barrier()
            # distribute saved model to other processes
            ckpt_file = os.path.join(self.output_path, "last_epoch_ckpt.pth")
            ckpt = torch.load(
                ckpt_file, map_location="cpu"
            )  # map_location={"cuda:0":"cuda:1"} not works
            self.model = load_ckpt(self.model, ckpt.get("model", {}))

    def test_model(self):
        # === Test ===#
        logger.info("Test start...")

        self.cfg.resume = False
        ckpt = os.path.join(self.output_path, "best_ckpt.pth")
        ckpt_dict = torch.load(ckpt, map_location="cpu")
        eval_ckpt = (
            ckpt_dict["model"]
            if ckpt_dict.get("model_ema", None) is None
            else ckpt_dict["model_ema"]
        )
        from autocare_dlt.core.utils.checkpoint import load_ckpt

        evalmodel = self.ema_model.ema if self.ema else self.model
        if is_parallel(evalmodel):
            evalmodel = evalmodel.module
        evalmodel = load_ckpt(evalmodel, eval_ckpt)

        evalmodel.eval()
        res, labels = self.inference(evalmodel, self.test_dataloader)
        if self.nms_thresh:
            res = nms(
                res, self.detections_per_img, self.nms_thresh, self.min_score
            )

        logger.info("Evaluate..")

        res = convert_to_coco_format(res, self.test_dataloader.dataset, labels)
        ap50_95, ap50, cls_ap_dict, summary = coco_evaluation(
            res,
            self.test_dataloader.dataset,
            ann_type="bbox",
            print_cls_ap=True,
        )
        tags = ["test/COCOAP50_95", "test/COCOAP50"]
        values = [ap50_95, ap50]
        for tag, x in cls_ap_dict.items():
            tags.append("test/" + tag)
            values.append(x)

        for tag, value in zip(tags, values):
            self.tblogger.add_scalar(tag, value, self.epoch)
        logger.info("\n" + summary)

    def evaluate_and_save_model(self):
        # === Evaluate ===#
        logger.info("Validation start...")

        evalmodel = self.ema_model.ema if self.ema else self.model
        if is_parallel(evalmodel):
            evalmodel = evalmodel.module

        evalmodel.eval()
        res, labels = self.inference(evalmodel, self.val_dataloader)
        if self.nms_thresh:
            res = nms(
                res, self.detections_per_img, self.nms_thresh, self.min_score
            )

        logger.info("Evaluate..")
        if res[0]["boxes"].size(dim=1) == 8:
            res_2point, res = convert_4pointBbox_to_coco_format(
                res, self.val_dataloader.dataset, labels
            )
            ap50_95, ap50, cls_ap_dict, summary = coco_evaluation(
                res_2point,
                self.val_dataloader.dataset,
                ann_type="bbox",
                print_cls_ap=self.verbose,
            )
        else:
            res = convert_to_coco_format(
                res, self.val_dataloader.dataset, labels
            )
            ap50_95, ap50, cls_ap_dict, summary = coco_evaluation(
                res,
                self.val_dataloader.dataset,
                ann_type="bbox",
                print_cls_ap=self.verbose,
            )

        # === Log ===#
        tags = ["val/COCOAP50_95", "val/COCOAP50"]
        values = [ap50_95, ap50]
        for tag, x in cls_ap_dict.items():
            tags.append("val/" + tag)
            values.append(x)

        for tag, value in zip(tags, values):
            self.tblogger.add_scalar(tag, value, self.epoch)

        logger.info("\n" + summary)

        # === Save Checkpoints ===#

        if self.target_cls + "/AP50_95" in cls_ap_dict:  # TODO AP50 지원
            target_ap = cls_ap_dict[self.target_cls + "/AP50_95"]
        else:
            target_ap = ap50_95

        self.save_ckpt(
            self.model,
            None if not self.ema else self.ema_model.ema,
            "last_epoch",
            target_ap > self.best_ap,
        )

        self.best_ap = max(self.best_ap, ap50_95)

    def before_iter(self):
        self.update_lr(self.progress_in_iter())

    def run_iter(self):
        loss = 0
        labeled_inputs, labeled_targets = self.labeled_iter()
        if torch.cuda.is_available():
            labeled_inputs = (
                labeled_inputs.cuda(self.rank)
                if self.distributed
                else labeled_inputs.cuda()
            )
            det_labels_to_cuda(labeled_targets, -1) if not (
                self.distributed
            ) else det_labels_to_cuda(labeled_targets, self.rank)

        sup_outputs = self.model(labeled_inputs)

        # loss
        sup_loss, sup_loss_dict = self.loss_manager(
            sup_outputs, labeled_targets
        )
        loss += sup_loss

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss_aver.update(loss)

        if self.ema:
            self.ema_model.update(self.model, iter=self.progress_in_iter())

        # log
        if self.iter % 100 == 0 or self.iter == 1:
            train_mesg = f"epoch {self.epoch} [{self.iter}/{self.iters_per_epoch}] - lr: {self.lr:0.6f} train loss: {self.loss_aver.avg:0.6f} "
            train_mesg += sup_loss_dict.to_string()
            if self.rank == 0:
                logger.info(train_mesg)

        # visualize every first iteration
        if self.cfg.get("DEBUG_data_sanity", False) and self.iter == 1:
            # TODO
            self.model.eval()
            sup_boxes, sup_scores, sup_labels = self.model(labeled_inputs)
            sup_outputs = [
                {"labels": sup_label, "boxes": sup_box, "scores": sup_score}
                for sup_box, sup_score, sup_label in zip(
                    sup_boxes, sup_scores, sup_labels
                )
            ]
            self.model.train()
            save_labels(
                labeled_inputs[0],
                labeled_targets[0],
                sup_outputs[0],
                save_path=os.path.join(self.output_path, "visualize"),
                prefix=self.epoch,
            )

    def after_iter(self):
        pass
