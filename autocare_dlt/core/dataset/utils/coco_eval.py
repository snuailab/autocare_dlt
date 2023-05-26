import contextlib
import io
import json
import tempfile

import numpy as np
import torch
from loguru import logger

try:
    from fast_coco_eval import COCOeval_fast as COCOeval
except (ImportError, ModuleNotFoundError):
    logger.warning(
        "Install fast-coco-eval is recommended, now using pycocotools."
    )
    from pycocotools.cocoeval import COCOeval


def coco_evaluation(results, dataset, ann_type="bbox", print_cls_ap=False):
    info = ""
    if len(results) > 0:
        coco_gt = dataset.coco
        _, tmp = tempfile.mkstemp()
        json.dump(results, open(tmp, "w"))
        cocoDt = coco_gt.loadRes(tmp)
        cocoEval = COCOeval(coco_gt, cocoDt, ann_type)
        cocoEval.params.imgIds = dataset.available_ids
        cocoEval.params.catIds = dataset.cls_mapping  ## debugging later
        cocoEval.evaluate()
        for key, v in cocoEval.eval.items():
            if isinstance(v, np.ndarray):
                v[np.isnan(v)] = 0

        cocoEval.accumulate()
        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            cocoEval.summarize()
        info += redirect_string.getvalue()
        ap_dicts, ap_string = cls_ap_summary(cocoEval.eval, coco_gt.cats)
        if print_cls_ap:
            info += ap_string
        return cocoEval.stats[0], cocoEval.stats[1], ap_dicts, info
    else:
        return 0, 0, {}, info


def convert_to_coco_format(outputs, dataset, labels=None):
    from autocare_dlt.core.utils import xyxy2xywh

    if labels is not None:
        labels = np.concatenate(labels)
    else:
        labels = [None] * len(outputs)
    cls_mapping = dataset.cls_mapping
    data_list = []
    img_ids = dataset.available_ids
    img_size = dataset.img_size
    img_w, img_h = (
        (img_size, img_size) if isinstance(img_size, int) else img_size
    )
    for i, output, lb in zip(img_ids, outputs, labels):
        img_info = dataset.coco.loadImgs(i)
        ori_width = img_info[0]["width"]
        ori_height = img_info[0]["height"]
        if lb is not None:
            (r_h, r_w), (pad_w, pad_h) = lb["ratio"]
            h0, w0 = lb["ori_shape"]
        else:
            (r_h, r_w), (pad_w, pad_h) = (1, 1), (0, 0)
            h0, w0 = ori_height, ori_width
        if output is None:
            continue
        bboxes = output["boxes"]
        bboxes[:, 0::2] = (
            (((bboxes[:, 0::2] * img_w) - pad_w) / r_w) / w0 * ori_width
        )
        bboxes[:, 1::2] = (
            (((bboxes[:, 1::2] * img_h) - pad_h) / r_h) / h0 * ori_height
        )
        bboxes = xyxy2xywh(bboxes)

        # preprocessing: resize
        if output["labels"] != None:
            labels = output["labels"].numpy()
        else:
            labels = None
        scores = output["scores"].numpy()
        for ind in range(bboxes.shape[0]):
            pred_data = {
                "image_id": int(i),
                "category_id": cls_mapping[labels[ind]]
                if labels is not None
                else None,
                "bbox": np.maximum(bboxes[ind], 0).numpy().tolist(),
                "score": float(scores[ind]),
            }  # COCO json format
            data_list.append(pred_data)
    return data_list


def convert_4pointBbox_to_coco_format(outputs, dataset, labels=None):
    if labels is not None:
        labels = np.concatenate(labels)
    else:
        labels = [None] * len(outputs)
    cls_mapping = dataset.cls_mapping
    data_list_2point = []
    data_list = []
    img_ids = dataset.available_ids
    img_size = dataset.img_size
    img_w, img_h = (
        (img_size, img_size) if isinstance(img_size, int) else img_size
    )
    for i, output, lb in zip(img_ids, outputs, labels):
        img_info = dataset.coco.loadImgs(i)
        ori_width = img_info[0]["width"]
        ori_height = img_info[0]["height"]
        (r_h, r_w), (pad_w, pad_h) = lb["ratio"]
        h0, w0 = lb["ori_shape"]
        if output is None:
            continue
        bboxes_2point = torch.zeros_like(output["boxes"][:, :4])
        bboxes = output["boxes"]
        bboxes_2point[:, 0::2] = (
            (((output["boxes"][:, [0, 4]] * img_w) - pad_w) / r_w)
            / w0
            * ori_width
        )
        bboxes_2point[:, 1::2] = (
            (((output["boxes"][:, [1, 5]] * img_h) - pad_h) / r_h)
            / h0
            * ori_height
        )
        bboxes[:, 0::2] = (
            (((output["boxes"][:, 0::2] * img_w) - pad_w) / r_w)
            / w0
            * ori_width
        )
        bboxes[:, 1::2] = (
            (((output["boxes"][:, 1::2] * img_h) - pad_h) / r_h)
            / h0
            * ori_height
        )
        bboxes_2point = xyxy2xywh(bboxes_2point)

        # preprocessing: resize
        labels = output["labels"].numpy()
        scores = output["scores"].numpy()
        for ind in range(bboxes_2point.shape[0]):
            pred_data_2point = {
                "image_id": int(i),
                "category_id": cls_mapping[labels[ind]],
                "bbox": np.maximum(bboxes_2point[ind], 0).numpy().tolist(),
                "score": float(scores[ind]),
            }  # COCO json format
            pred_data = {
                "image_id": int(i),
                "category_id": cls_mapping[labels[ind]],
                "bbox": np.maximum(bboxes[ind], 0).numpy().tolist(),
                "score": float(scores[ind]),
            }  # COCO json format
            data_list_2point.append(pred_data_2point)
            data_list.append(pred_data)
    return data_list_2point, data_list


def cls_ap_summary(eval, cats):
    params = eval["params"]
    prec = eval["precision"]
    ap_dict = {}
    ap_string = " AP50:95 per classes\n"
    for i, cat_id in enumerate(params.catIds):
        cls_prec = prec[:, :, i, 0, 2]
        cls_ap = np.mean(cls_prec[cls_prec > -1])
        cls_ap = 0 if np.isnan(cls_ap) else cls_ap
        cls_key = cats[cat_id]["name"] + "/AP50_95"
        ap_dict.update({cls_key: cls_ap})
        if i % 4 == 3:
            ap_string += f" {cats[cat_id]['name']:<20}{cls_ap:.6f}\n"
        else:
            ap_string += f" {cats[cat_id]['name']:<20}{cls_ap:.6f},"
    ap_string += "\n"
    ap_string += " AP50 per classes\n"
    for i, cat_id in enumerate(params.catIds):
        cls_prec = prec[0, :, i, 0, 2]
        cls_ap = np.mean(cls_prec[cls_prec > -1])
        cls_ap = 0 if np.isnan(cls_ap) else cls_ap
        cls_key = cats[cat_id]["name"] + "/AP50"
        ap_dict.update({cls_key: cls_ap})
        if i % 4 == 3:
            ap_string += f" {cats[cat_id]['name']:<20}{cls_ap:.6f}\n"
        else:
            ap_string += f" {cats[cat_id]['name']:<20}{cls_ap:.6f},"
    return ap_dict, ap_string
