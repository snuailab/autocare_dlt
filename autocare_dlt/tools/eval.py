import argparse
import os
import sys
from typing import Union

import numpy as np
import torch
from box import Box
from loguru import logger

sys.path.append(os.getcwd())
from autocare_dlt.core.dataset import *
from autocare_dlt.core.dataset.utils import *
from autocare_dlt.core.model import build_model
from autocare_dlt.core.utils import Inferece
from autocare_dlt.utils.config import *


def make_parser():
    parser = argparse.ArgumentParser("Tx evaluator")

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
    return parser


def run(
    model_cfg: str,
    data_cfg: str,
    gpus: str,
    ckpt: Union[str, dict],
) -> None:
    """Evaluate a model

    Args:
        model_cfg (str): path for model configuration file
        data_cfg (str): path for dataset configureation file
        gpus (str): GPU IDs to use
        ckpt (Union[str, dict]): path for checkpoint file or state dict
    """

    args = Box(
        {
            "model_cfg": model_cfg,
            "data_cfg": data_cfg,
            "gpus": gpus,
            "ckpt": ckpt,
        }
    )

    cfg = parsing_config(args)

    if cfg.gpus != "-1" and torch.cuda.is_available():
        torch.cuda.set_device(device=f"cuda:{cfg.gpus}")
    else:
        torch.cuda.is_available = lambda: False

    logger.info("Building Model for evaluation")
    model, classes = build_model(cfg, strict=True)
    model.eval()

    logger.info("Building Dataset for evaluation")
    cfg["data"]["task"] = cfg.get("task", None)
    cfg["data"]["classes"] = (
        classes[0] if cfg.task in classifier_list else classes
    )
    dataset = getattr(sys.modules[__name__], cfg.data.test.type)(
        cfg.data, cfg.data.test
    )

    logger.info("Run Model..")
    inferencer = Inferece(cfg)
    if torch.cuda.is_available():
        model.cuda()

    if cfg.task in detector_list:
        from autocare_dlt.core.dataset.utils import (
            coco_evaluation,
            collate_fn,
            convert_4pointBbox_to_coco_format,
            convert_to_coco_format,
        )
        from autocare_dlt.core.utils import nms

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.data.batch_size_per_gpu,
            num_workers=cfg.data.workers_per_gpu,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        res, labels = inferencer(model.eval(), dataloader)

        logger.info("Post Processing..")
        nms_thresh = cfg.get("nms_thresh", 0.5)
        detections_per_img = cfg.get("detections_per_img", 100)
        min_score = cfg.get("min_score", 0.01)
        res = nms(res, detections_per_img, nms_thresh, min_score)

        logger.info("Evaluate..")
        if res[0]["boxes"].size(dim=1) == 8:
            res, _ = convert_4pointBbox_to_coco_format(res, dataset, labels)
        else:
            res = convert_to_coco_format(res, dataset, labels)
        ap50_95, ap50, cls_ap_dict, summary = coco_evaluation(
            res, dataset, ann_type="bbox", print_cls_ap=True
        )
        logger.info("\n" + summary)
    elif cfg.task in classifier_list:

        def collate_fn(datas):
            imgs, labels = zip(*datas)
            return torch.stack(imgs), torch.stack(labels).transpose(0, 1)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.data.batch_size_per_gpu,
            num_workers=cfg.data.workers_per_gpu,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        outputss, targetss = inferencer(model.eval(), dataloader)

        logger.info("Evaluate..")
        # log metrics
        score_texts = []
        metrics = cfg.get(
            "eval_metrics", ["accuracy", "precision", "recall", "f1"]
        )
        res = {met: [] for met in metrics}
        for outputs, targets in zip(outputss, targetss):
            for output, target in zip(outputs, targets):
                values = cls_eval(output.detach().cpu(), target.detach().cpu())
                for metric in metrics:
                    res[metric].append(values[metric])
        score_texts = []
        values = []
        for method in metrics:
            metric = res[method]
            # logger
            if isinstance(metric, list):
                score_str = f"{method:<20}"
                # TODO cls_eval is not working for multi cls..?
                # for attr, s in zip(self.classes, metric):
                #     score_str += f'{attr}: {s}, '
                score_str += str(np.mean(metric))
                score_texts.append(score_str)
                metric = np.mean(metric)
            else:
                score_texts.append(f"{method:<20}{metric:.6f}")

        score_text = "\n".join(score_texts)
        logger.info("Evaluation scores\n" + score_text)
    elif cfg.task in regressor_list:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.data.batch_size_per_gpu,
            num_workers=cfg.data.workers_per_gpu,
            shuffle=False,
            pin_memory=True,
        )
        preds, labels = inferencer(model.eval(), dataloader)

        logger.info("Evaluate..")
        labels = torch.cat(labels, dim=0)
        metrics = cfg.get("eval_metrics", ["mae", "mse", "rmse"])
        if torch.is_tensor(preds[0]):
            preds = torch.cat(preds).view(-1, cfg.num_classes)  # (labels)
            labels = labels.view(-1)
            res = reg_eval(preds, labels)
        else:
            res = multi_attr_eval(cfg.classes, preds, labels)
        for method in metrics:
            metric = res[method]
            # logger
            if isinstance(metric, list):
                score_str = f"{method:<20}"
                for attr, s in zip(cfg.classes, metric):
                    score_str += f"{attr}: {s}, "
                score_texts.append(score_str)
                metric = np.mean(metric)
            else:
                score_texts.append(f"{method:<20}{metric:.6f}")
        score_text = "\n".join(score_texts)
        logger.info("Evaluation scores\n" + score_text)

        # TODO: need to add multi attribute code. this is for single attribute.
        logger.info("Evaluate..")
        model.train()
        for pred, gt in zip(preds[:5], labels[:5]):
            if "Attention" in model.stages["Pred"]:
                pred = pred[: pred.find("[s]")]
                gt = gt[: gt.find("[s]")]
            print(f"{pred:20s}, gt: {gt:20s},   {str(pred == gt)}")
        valid_log = f"Test accuracy: {current_accuracy:0.3f}, Test norm_ED: {current_norm_ED:0.2f}"
        print(valid_log)
    elif cfg.task in str_list:

        def collate_fn(datas):
            imgs, labels = zip(*datas)
            return torch.stack(imgs), torch.stack(labels)

        idx2char = {}
        idx2char[0] = ""
        for idx, char in enumerate(cfg.classes, start=1):
            idx2char[idx] = char
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.data.batch_size_per_gpu,
            num_workers=cfg.data.workers_per_gpu,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        with torch.no_grad():
            outputss, targetss = inferencer(model.eval(), dataloader)
        logger.info("Evaluate..")
        metrics = cfg.get("eval_metrics", ["accuracy", "norm_ED"])

        res = {metric: [] for metric in metrics}
        for outputs, targets in zip(outputss, targetss):
            outputs = torch.argmax(outputs, dim=-1)
            outputs, targets = decoder(outputs, idx2char), decoder(
                targets, idx2char
            )
            for output, target in zip(outputs, targets):
                values = str_eval(output, target)
                for metric in metrics:
                    res[metric].append(values[metric])
        score_texts = []
        for method in metrics:
            metric = res[method]
            # logger
            if isinstance(metric, list):
                metric = np.mean(metric)
            score_texts.append(f"{method:<20}{metric:.6f}")
        score_text = "\n".join(score_texts)
        logger.info("Evaluation scores\n" + score_text)

    elif cfg.task in pose_estimator_list:
        from autocare_dlt.core.dataset.utils import (
            collate_fn,
        )
        from autocare_dlt.core.dataset.utils.pose_eval import pck_accuracy

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.data.batch_size_per_gpu,
            num_workers=cfg.data.workers_per_gpu,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        res, labels = inferencer(model.eval(), dataloader)
        res = torch.cat(res).numpy()
        labels = np.concatenate(labels).tolist()
        _, score, cnt, pred = pck_accuracy(
            res,
            labels,
        )
        logger.info(f"Test PCK Score: {score}")
    else:
        raise KeyError(f"cfg.task: {cfg.task} is unsupported task.")


if __name__ == "__main__":
    args = make_parser().parse_args()
    run(**vars(args))
