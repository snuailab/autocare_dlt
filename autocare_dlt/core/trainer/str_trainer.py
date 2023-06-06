import os
import sys

import numpy as np
import torch
from loguru import logger
from torch.utils.data.distributed import DistributedSampler

from autocare_dlt.core.dataset.utils import (
    DataIterator,
    decoder,
    str_eval,
)
from autocare_dlt.core.model.utils.functions import is_parallel
from autocare_dlt.core.trainer import BaseTrainer
from autocare_dlt.core.utils.functions import check_gpu_availability


class StrTrainer(BaseTrainer):
    def __init__(self, model, datasets, cfg):

        super().__init__(model, datasets, cfg)

        # === Trainer Configs === #
        self.metrics = self.cfg.get("eval_metrics", ["accuracy", "norm_ED"])

        self.idx2char = {}
        self.idx2char[0] = ""
        for idx, char in enumerate(self.cfg.classes, start=1):
            self.idx2char[idx] = char

        # === Values === #
        self.best_score = -1

    def _get_dataloader(self):
        def collate_fn(datas):
            imgs, labels = zip(*datas)
            return torch.stack(imgs), torch.stack(labels)

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
        # dist.barrier()
        self._get_lr_scheduler(self.start_lr, self.iters_per_epoch)

        # === GPU memory availability check === #
        if torch.cuda.is_available():
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

            self.cuda()

    def after_train(self):
        logger.info(
            f"Training of experiment is done and the best {self.metrics[0]} is {self.best_score * 100:.2f}"
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
        # === Test ===#
        logger.info("Test start...")

        self.cfg.resume = False
        self.cfg.ckpt = os.path.join(self.output_path, "best_ckpt.pth")
        self.resume_train()

        evalmodel = self.model
        if is_parallel(evalmodel):
            evalmodel = evalmodel.module

        evalmodel.eval()
        with torch.no_grad():
            outputss, targetss = self.inference(evalmodel, self.val_dataloader)
        logger.info("Evaluate..")

        res = {metric: [] for metric in self.metrics}
        for outputs, targets in zip(outputss, targetss):
            outputs = torch.argmax(outputs, dim=-1)
            outputs, targets = decoder(outputs, self.idx2char), decoder(
                targets, self.idx2char
            )
            for output, target in zip(outputs, targets):
                values = str_eval(output, target)
                for metric in self.metrics:
                    res[metric].append(values[metric])

        # === Log ===#
        score_texts = []
        tags = []
        values = []
        for method in self.metrics:
            metric = res[method]
            # logger
            if isinstance(metric, list):
                metric = np.mean(metric)
            score_texts.append(f"{method:<20}{metric:.6f}")

            tags.append(f"test/{method}")
            values.append(metric)

        for tag, value in zip(tags, values):
            self.tblogger.add_scalar(tag, value, self.epoch)

        score_text = "\n".join(score_texts)
        logger.info("Test scores\n" + score_text)

    def evaluate_and_save_model(self):
        # === Evaluate ===#
        logger.info("Validation start...")

        evalmodel = self.ema_model.ema if self.ema else self.model
        if is_parallel(evalmodel):
            evalmodel = evalmodel.module

        evalmodel.eval()
        with torch.no_grad():
            outputss, targetss = self.inference(evalmodel, self.val_dataloader)
        logger.info("Evaluate..")

        res = {metric: [] for metric in self.metrics}
        score_texts = []
        for i, (outputs, targets) in enumerate(zip(outputss, targetss)):
            outputs = torch.argmax(outputs, dim=-1)
            outputs, targets = decoder(outputs, self.idx2char), decoder(
                targets, self.idx2char
            )
            for output, target in zip(outputs, targets):
                values = str_eval(output, target)
                for metric in self.metrics:
                    res[metric].append(values[metric])
            if i % 100 == 0:
                score_texts.append(f"pred, target: {output}, {target}")
        logger.info("SAMPLE\n" + "\n".join(score_texts))
        score = np.mean(res[self.metrics[0]])

        # === Log ===#
        score_texts = []
        tags = []
        values = []
        for method in self.metrics:
            metric = res[method]
            # logger
            if isinstance(metric, list):
                metric = np.mean(metric)
            score_texts.append(f"{method:<20}{metric:.6f}")

            tags.append(f"val/{method}")
            values.append(metric)

        for tag, value in zip(tags, values):
            self.tblogger.add_scalar(tag, value, self.epoch)

        score_text = "\n".join(score_texts)
        logger.info("Validation scores\n" + score_text)

        # === Save Checkpoints ===#
        self.save_ckpt(
            self.model,
            self.ema_model.ema if self.ema else None,
            "last_epoch",
            score > self.best_score,
        )

        self.best_score = max(self.best_score, score)

    def before_iter(self):
        self.update_lr(self.progress_in_iter())

    def run_iter(self):
        loss = 0
        labeled_inputs, labeled_targets = self.labeled_iter()
        if torch.cuda.is_available():
            labeled_inputs = labeled_inputs.cuda()
            labeled_targets = labeled_targets.cuda()

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
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5, error_if_nonfinite=True)
        self.loss_aver.update(loss)

        outputs = torch.argmax(sup_outputs, dim=-1)
        outputs, targets = decoder(outputs, self.idx2char), decoder(
            labeled_targets, self.idx2char
        )
        for output, target in zip(outputs, targets):
            self.acc_aver.update(str_eval(output, target)["accuracy"])

        if self.ema:
            self.ema_model.update(self.model, iter=self.progress_in_iter())

        # log
        if self.iter % 100 == 0 or self.iter == 1:
            train_mesg = f"epoch {self.epoch} [{self.iter}/{self.iters_per_epoch}] - lr: {self.lr:0.6f} train loss: {self.loss_aver.avg:0.6f} "
            train_mesg += sup_loss_dict.to_string()
            logger.info(train_mesg)

        # visualize every first iteration
        if self.cfg.get("DEBUG_data_sanity", False) and self.iter == 1:
            self.model.eval()
            self.model.train()

    def after_iter(self):
        pass
