import os
import os
import sys

import torch
import torch.distributed as dist
import numpy as np
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from autocare_dlt.core.dataset.utils import (
    DataIterator,
    seg_evaluation,
    collate_fn
)
from autocare_dlt.core.model.utils.functions import is_parallel
from autocare_dlt.core.trainer import BaseTrainer
from autocare_dlt.core.utils import (
    seg_labels_to_cuda,
    load_ckpt,
    nms,
)
from autocare_dlt.core.utils.functions import check_gpu_availability
from autocare_dlt.utils.debugging import save_labels
from autocare_dlt.utils.visualization import log_graph


class SegmentationTrainer(BaseTrainer):
    def __init__(self, model, datasets, cfg):

            super().__init__(model, datasets, cfg)
    
            self.best_loss = 10000000000        
            self.train_loss_history = []
            self.val_loss_history = []
            self.gray = cfg["data"].get("gray", False)

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

        # if self.rank == 0 and torch.cuda.is_available():
        #     gpu_availability = check_gpu_availability(
        #         model=self.model,
        #         input_size=self.cfg.data.img_size,
        #         batch_size=self.cfg.data.batch_size_per_gpu,
        #         dtype=self.data_type,
        #         gpu_total_mem=torch.cuda.get_device_properties(
        #             0
        #         ).total_memory,  # Bytes
        #     )  # Bool. # TODO: Consider allocated memory

        #     if not gpu_availability:
        #         sys.exit(-1)

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
                f"Training of experiment is done and the best validation loss is {self.best_loss:.5f}"
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

        # TODO: remove
        self.train_loss_history.append(self.loss_aver.avg)

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

        logger.info("Evaluate..")

        _, mpa, recalls, precisions = seg_evaluation(res, labels, self.cfg.classes, self.loss_manager)
        
        tags = ["test/mpa"]
        values = [mpa]

        for tag, value in zip(tags, values):
            self.tblogger.add_scalar(tag, value, self.epoch)

        summary = f"mean pixel accuracy: {mpa:.5f}"
        for c in list(recalls.keys()):
            summary += f"\n{c} - recall: {recalls[c]:.5f}, precision: {precisions[c]:.5f}"
        logger.info(summary)

    def evaluate_and_save_model(self):
        # === Evaluate ===#
        logger.info("Validation start...")

        evalmodel = self.ema_model.ema if self.ema else self.model
        if is_parallel(evalmodel):
            evalmodel = evalmodel.module

        evalmodel.eval()
        res, labels = self.inference(evalmodel, self.val_dataloader)
        
        logger.info("Evaluate..")
   
        val_loss, mpa, recalls, precisions = seg_evaluation(res, labels, self.cfg.classes, self.loss_manager)
        
        # TODO: remove
        self.val_loss_history.append(val_loss)


        # === Log ===#
        tags = ["val/loss", "val/mpa"]
        values = [val_loss, mpa]

        for tag, value in zip(tags, values):
            self.tblogger.add_scalar(tag, value, self.epoch)

        summary = f"\nval loss: {val_loss:.5f}\nmean pixel accuracy: {mpa:.5f}"
        for c in list(recalls.keys()):
            summary += f"\n{c} - recall: {recalls[c]:.5f}, precision: {precisions[c]:.5f}"
        logger.info(summary)

        # === Save Checkpoints ===#
        self.save_ckpt(
            self.model,
            None if not self.ema else self.ema_model.ema,
            "last_epoch",
            val_loss < self.best_loss,
        )

        self.best_loss = min(self.best_loss, val_loss)

        # TODO: remove
        log_graph(self.train_loss_history, self.val_loss_history, "Loss", self.output_path)

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
            seg_labels_to_cuda(labeled_targets, -1) if not (
                self.distributed
            ) else seg_labels_to_cuda(labeled_targets, self.rank)

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
        
        # visualize every first training example
        # TODO: move to somewhere
        if self.epoch == 0 and self.iter == 1:
            sample_image = labeled_inputs[0]
            sample_mask = labeled_targets[0]["labels"]

            cmap = np.zeros((sample_mask.shape[0], sample_mask.shape[1], 3))
            colors = np.random.randint(255, size=(len(self.cfg.classes), 3))
            for cls in range(len(self.cfg.classes)):
                loc = 1*(sample_mask == cls).cpu().numpy()
                cmap_temp = []
                for i in range(3):
                    cmap_temp.append(np.expand_dims(loc*colors[cls][i], axis=2))
                cmap_temp = np.concatenate(cmap_temp, axis=2)
                cmap = cmap + cmap_temp

            sample_image = 255*sample_image.cpu().numpy().transpose(1, 2, 0)
            if self.gray:
                sample_image = np.concatenate((sample_image, sample_image, sample_image), axis=2)

            import cv2 
            visual_sample = cv2.addWeighted(sample_image.astype(np.float64), 0.5, cmap, 0.5, 0)
            cv2.imwrite(f"{self.output_path}/training_sample.png", visual_sample)

    def after_iter(self):
        pass
