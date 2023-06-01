from functools import partial

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from autocare_dlt.core.dataset.utils import letterbox
from autocare_dlt.core.dataset.utils.pose_eval import get_max_preds
from autocare_dlt.core.utils import nms, xyxy2xywh
from autocare_dlt.utils.config import (
    classifier_list,
    detector_list,
    pose_estimator_list,
    regressor_list,
    str_list,
    segmenter_list
)


class Inferece:
    def __init__(self, cfg, single_img=False):
        self.cfg = cfg
        self.single_img = single_img
        self.postprocessing = self.build_postprocess(cfg, single_img)
        self.preprocessing = self.build_preporcess(cfg) if single_img else None

    def __call__(self, model, inputs):
        """

        ``self.preprocessing()``:
            It resizes the input img to the arg ``--input_size``.
            The default is ``[640]``

        Args:
            inputs (np.ndarray):
                shape: [height, width, channel(BGR)]
        """
        model.eval()
        if self.single_img:
            # Run with infererece.py
            inputs, meta = self.preprocessing(inputs)  # ndarray to tensor
            out = model(
                inputs
            )  # inputs shape: [batch_size, channel, height, width]
            return self.postprocessing(
                out, meta
            )  # out shape: [batch_size, num_joints, height, width]
        else:
            # Run in trainers or eval
            outs = []
            labels = []

            for img, label in inputs:  # datadloader
                if isinstance(img, list):  # In case of weak strong img pair
                    img = img[0]

                if torch.cuda.is_available():
                    img = img.cuda()
                out = model(img)
                # Detection case
                # out(tuple) = bbox(torch.tensor) (batch size, num detection, 4), scores(torch.tensor) (batch size, num detection, 1), labels(torch.tensor) (batch size, num detection, 1)
                # Classification case
                # out(torch.tensor) (batch size, num classes, 1, 1)
                outs.append(out)
                labels.append(label)

            return (
                outs,
                labels,
            )  # TODO add postprocessing after refactoring Data I/O...

    def build_preporcess(self, cfg):
        input_size = cfg["input_size"]
        letter_box = cfg.get("letter_box", False)
        if letter_box:
            return LetterBoxPreprocess(input_size)
        else:
            return SimplePreprocess(input_size)

    def build_postprocess(self, cfg, single_img):
        task = cfg.get("task", None)
        if task in detector_list:
            return DetPostProcess(cfg, single_img)
        elif task in classifier_list:
            return ClsPostProcess(cfg, num_classes=cfg.get("num_classes", 1))
        elif task in regressor_list:
            return ClsPostProcess(cfg, num_classes=1)
        elif task in str_list:
            return STRPostProcess(cfg)
        elif task in pose_estimator_list:
            return PosePostProcess(cfg)
        elif task in segmenter_list:
            return SegPostProcess(cfg)
        else:
            raise ValueError(f"cfg.task: {task} is unsupported task.")


## TODO: could be moved to somewhere else...
class DetPostProcess:
    def __init__(self, cfg, single_img):
        self.detections_per_img = cfg.get("detections_per_img", 100)
        self.nms_thresh = cfg.get("nms_thresh", 0.5)
        self.min_score = cfg.get("min_score", 0.01)
        input_size = (
            cfg["input_size"] if single_img else cfg["data"]["img_size"]
        )
        if isinstance(input_size, int):
            self.input_size = [input_size, input_size]
        else:
            if len(input_size) > 1:
                self.input_size = input_size
            else:
                self.input_size = [input_size[0], input_size[0]]

            self.single_img = single_img

    def __call__(self, input, meta):
        data_list = []

        if self.single_img:
            input = [input]
            meta = [meta]

        output = nms(
            input, self.detections_per_img, self.nms_thresh, self.min_score
        )
        for ot, mt in zip(output, meta):
            r_h, r_w = mt["ratio"]
            pad_w, pad_h = mt["pad"]
            bboxes = ot["boxes"]
            if len(bboxes) > 0:
                bboxes[:, 0::2] = (
                    (bboxes[:, 0::2] * self.input_size[0]) - pad_w
                ) / r_w
                bboxes[:, 1::2] = (
                    (bboxes[:, 1::2] * self.input_size[1]) - pad_h
                ) / r_h
                if len(bboxes[0]) == 4:
                    bboxes = xyxy2xywh(bboxes).numpy()
                else:
                    bboxes = bboxes.numpy()
                # preprocessing: resize
                labels = ot["labels"].numpy()
                scores = ot["scores"].numpy()

                for ind in range(bboxes.shape[0]):
                    pred_data = {
                        "category_id": int(labels[ind]) + 1,
                        "bbox": np.maximum(bboxes[ind], 0).tolist(),
                        "score": float(scores[ind]),
                        "area": float(bboxes[ind][2] * bboxes[ind][3]),
                        "iscrowd": 0,
                    }  # COCO json format
                    data_list.append(pred_data)
        return data_list


class ClsPostProcess:
    def __init__(self, cfg, num_classes):
        self.num_classes = num_classes
        self.classes = list(cfg.classes[0].keys())

    def __call__(self, input, meta):
        # meta is dummy input now
        if isinstance(input, list):
            # eval
            data_list = []
            cate_id = 1
            for attr_id, attr in enumerate(input):
                for batch_index, cls in enumerate(attr):
                    score, cls_idx = torch.max(cls, dim=0)
                    pred_data = {
                        "category_id": cate_id + int(cls_idx),
                        "score": float(score),
                    }
                    data_list.append(pred_data)
                cate_id += len(cls)
            return data_list
        else:
            # inference
            return [input.view(-1, self.num_classes)]


class STRPostProcess:
    def __init__(self, cfg):
        self.classes = cfg.classes
        self.idx2char = {}
        self.idx2char[0] = ""
        for idx, char in enumerate(self.classes, start=1):
            self.idx2char[idx] = char

    def __call__(self, input, meta):
        data_list = []

        text = []
        for pred in input:
            score, pred = torch.max(F.softmax(pred, dim=-1), dim=-1)

            for p in pred.tolist():
                if p != 0:
                    text.append(self.idx2char[p])
            text = "".join(text)

            pred_data = {
                "caption": text,
                "score": float(torch.mean(score)),
                "category_id": -1,
            }
            data_list.append(pred_data)

        return data_list


class PosePostProcess:
    def __init__(self, cfg):
        pass

    def __call__(self, input: torch.Tensor, meta: dict) -> list:
        """It converts the model's raw output to keypoints.

        It extracts the keypoints coordinates from the heatmap.

        ``raw`` represents the input img before being resized.
        ``in`` represents the resized input img as the model input.
        ``out`` represents the output(heatmaps) of the model.

        Args:
            input (torch.Tensor): Model's raw output. (Joint heatmaps.)
                shape: [1(batch_size), num_joints, target_height, target_width]

            meta (dict):
                ratio:
                    Ratio for <raw img -> resized input img>.
                    (height_ratio, width_ratio)

                ori_shape: Raw img height and width.

        Returns:
            keypoints (list): Keypoints coords scaled to the raw img.
                shape: [num_joints, 2(x and y)]
        """

        batch_heatmaps_tensor = input
        raw2in_height_ratio, raw2in_width_ratio = (
            meta.get("ratio")[0],
            meta.get("ratio")[1],
        )
        raw_height, raw_width = (
            meta.get("ori_shape")[0],
            meta.get("ori_shape")[1],
        )
        out_height, out_width = (
            batch_heatmaps_tensor.shape[2],
            batch_heatmaps_tensor.shape[3],
        )
        in_height, in_width = (
            raw_height * raw2in_height_ratio,
            raw_width * raw2in_width_ratio,
        )

        out2in_height_ratio, out2in_width_ratio = (
            in_height / out_height,
            in_width / out_width,
        )
        in2raw_height_ratio, in2raw_width_ratio = (
            1 / raw2in_height_ratio,
            1 / raw2in_width_ratio,
        )

        out2raw_height_ratio, out2raw_width_ratio = (
            out2in_height_ratio * in2raw_height_ratio,
            out2in_width_ratio * in2raw_width_ratio,
        )

        batch_keypoints, _ = get_max_preds(batch_heatmaps_tensor.cpu().numpy())
        keypoints = batch_keypoints[0]
        keypoints_x, keypoints_y = keypoints[:, [0]], keypoints[:, [1]]
        resized_keypoints_x = keypoints_x * out2raw_width_ratio
        resized_keypoints_y = keypoints_y * out2raw_height_ratio
        resized_keypoints = np.concatenate(
            (resized_keypoints_x, resized_keypoints_y), axis=1
        )

        return resized_keypoints.tolist()

# TODO: SegPostProcess for inference(should be project specific?)
class SegPostProcess:
    def __init__(self, cfg):
        self.classes = list(cfg.classes)

    def __call__(self, input, meta):
        # meta is dummy input now
        if isinstance(input, list):
            return 0
        

class LetterBoxPreprocess:
    def __init__(self, input_size):
        if len(input_size) > 1:
            raise ValueError("letter box requires single input_size values")
        self.input_size = input_size[0]

    def __call__(self, input):
        h0, w0 = input.shape[:2]
        input, ratio, pad = letterbox(
            input, self.input_size, auto=False, scaleup=False
        )
        meta = {"ratio": ratio, "pad": pad, "ori_shape": (h0, w0)}
        return img2tensor(input), meta


class SimplePreprocess:
    def __init__(self, input_size):
        self.input_size = input_size
        if len(self.input_size) > 1:
            self.w, self.h = self.input_size[0], self.input_size[1]
        else:
            self.w, self.h = self.input_size[0], self.input_size[0]

    def __call__(self, input):
        h0, w0 = input.shape[:2]

        ratio = (self.h / h0, self.w / w0)
        input = cv2.resize(
            input, (self.w, self.h), interpolation=cv2.INTER_AREA
        )
        meta = {"ratio": ratio, "pad": (0, 0), "ori_shape": (h0, w0)}
        return img2tensor(input), meta


def img2tensor(img):
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).unsqueeze(0) / 255
    if torch.cuda.is_available():
        img = img.cuda()
    return img
