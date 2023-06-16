import argparse
import json
import os
import sys
from typing import Union

import cv2
import torch
import tqdm
from box import Box
from loguru import logger
from pycocotools.coco import COCO

sys.path.append(os.getcwd())
from autocare_dlt.core.model import build_model
from autocare_dlt.core.utils import Inferece
from autocare_dlt.utils.config import parsing_config, str2bool
from autocare_dlt.utils.visualization import DrawResults


def arange_inputs(input_path):
    if os.path.isdir(input_path):
        input = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.split(".")[-1] in ["jpg", "png", "bmp"]
        ]
    else:
        if input_path.split(".")[-1] in ["jpg", "png", "bmp"]:
            input = [input_path]
        elif input_path.split(".")[-1] in ["avi", "mp4"]:
            input = input_path
        elif input_path.split(".")[-1] in ["json"]:
            input = COCO(input_path)
        else:
            raise ValueError()
    return input


def make_parser():
    parser = argparse.ArgumentParser("Tx evaluator")

    parser.add_argument(
        "--inputs",
        default=None,
        type=str,
        help="input path - img, dir or json",
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
        "--output_dir",
        type=str,
        help="path for inference results",
        required=True,
    )
    parser.add_argument(
        "--root_dir", type=str, default="", help="root_dir for coco image"
    )
    parser.add_argument(
        "--gpus", type=str, default="0", help="gpus for training"
    )
    parser.add_argument(
        "--ckpt", default=None, type=str, help="checkpoint file"
    )
    parser.add_argument(
        "--input_size",
        default=[640],
        nargs="+",
        type=int,
        help="input size of model inference",
    )
    parser.add_argument(
        "--letter_box", default=False, type=str2bool, help="use letter box"
    )
    parser.add_argument(
        "--vis",
        default=False,
        type=str2bool,
        help="visualize inference in realtime",
    )
    parser.add_argument(
        "--save_imgs",
        default=False,
        type=str2bool,
        help="draw and save inference results as images",
    )
    parser.add_argument(
        "--save_video",
        default=False,
        type=str2bool,
        help="draw and save inference results as a video",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        type=str2bool,
        help="overwrite former results",
    )
    parser.add_argument(
        "--gray",
        default=False,
        type=str2bool,
        help="gray scale",
    )

    return parser


def run(
    inputs: str,
    model_cfg: str,
    output_dir: str,
    gpus: str,
    ckpt: Union[str, dict],
    input_size: list = None,
    letter_box: bool = None,
    vis: bool = False,
    save_imgs: bool = False,
    save_video: bool = False,
    root_dir: str = "",
    overwrite: bool = False,
    gray: bool = False
) -> None:
    """Run inference

    Args:
        inputs (str): path for input - image, directory, or json
        model_cfg (str): path for model configuration file
        output_dir (str): path for inference results
        gpus (str): GPU IDs to use
        ckpt (Union[str, dict]): path for checkpoint file or state dict
        input_size (list, optional): input size of model inference. Defaults to [640].
        letter_box (bool, optional): whether to use letter box or not. Defaults to False.
        vis (bool, optional): whether to visualize inference in realtime or not. Defaults to False.
        save_imgs (bool, optional): whether to draw and save inference results as images or not. Defaults to False.
        save_video (bool, optional): whether to draw and save inference results as a video or not. Defaults to False.
        root_dir (str, optional): path for input image when using json input. Defaults to "".
    """

    args = Box(
        {
            "inputs": inputs,
            "model_cfg": model_cfg,
            "output_dir": output_dir,
            "gpus": gpus,
            "ckpt": ckpt,
            "input_size": input_size,
            "letter_box": letter_box,
            "vis": vis,
            "save_imgs": save_imgs,
            "save_video": save_video,
            "overwrite": overwrite,
            "gray": gray
        }
    )

    cfg = parsing_config(args)

    if cfg.gpus != "-1" and torch.cuda.is_available():
        torch.cuda.set_device(device=f"cuda:{cfg.gpus}")
    else:
        torch.cuda.is_available = lambda: False

    logger.info("Create result saving folder...")
    os.makedirs(cfg.output_dir, exist_ok=cfg.overwrite)

    logger.info("Build Model..")
    model, classes = build_model(cfg, strict=True)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    if cfg.vis or cfg.save_imgs or cfg.save_video:
        font_path = cfg.get("font_path", False)
        draw = DrawResults(cfg.task, classes, font_path)
        if cfg.save_imgs:
            os.mkdir(f"{cfg.output_dir}/imgs")
        if cfg.save_video:
            os.mkdir(f"{cfg.output_dir}/video")

    else:
        draw = False

    logger.info("Setup inferece..")
    input = arange_inputs(cfg.inputs)
    inference = Inferece(cfg, single_img=True)

    logger.info("Run inference..")
    outputs = {"categories": [], "images": [], "annotations": []}
    if isinstance(classes[0], dict):  # if use attribute
        cat_id = 0
        for attr, cats in classes[0].items():
            for i, cat_name in enumerate(cats, start=1):
                outputs["categories"].append(
                    {"id": cat_id + i, "name": cat_name, "supercategory": attr}
                )
            cat_id += len(cats)
    else:
        for cat_id, cat_name in enumerate(classes, start=1):
            outputs["categories"].append(
                {"id": cat_id, "name": cat_name, "supercategory": cat_name}
            )

    ann_id = 1

    img_paths = []
    img_ids = []
    if isinstance(input, COCO):
        img_paths = [img_info["file_name"] for img_info in input.imgs.values()]
        img_dirnames = set(map(os.path.dirname, img_paths))
        for img_dirname in img_dirnames:
            os.makedirs(f"{cfg.output_dir}/imgs/{img_dirname}")
        img_ids = [img_info["id"] for img_info in input.imgs.values()]
        input = [
            os.path.join(root_dir, img_info["file_name"])
            for img_info in input.imgs.values()
        ]
    if isinstance(input, list):  # image or dir
        pbar = tqdm.tqdm(input)
        if len(img_ids) < 1:
            img_ids = [i for i in range(1, len(input) + 1)]
        for idx, (img_id, img_path) in enumerate(zip(img_ids, pbar)):
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            
            if cfg.gray:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img_name = (
                img_paths[idx] if img_paths else os.path.split(img_path)[-1]
            )
            res = inference(model, img)
            if draw:
                img = draw.run(img, res)
                if cfg.vis:
                    cv2.imshow("visualization", img)
                    cv2.waitKey(1)
                if cfg.save_imgs:
                    cv2.imwrite(f"{cfg.output_dir}/imgs/{img_name}", img)

            outputs["images"].append(
                {"id": img_id, "file_name": img_name, "width": w, "height": h}
            )
            for ann in res:
                ann.update({"id": ann_id, "image_id": img_id})
                outputs["annotations"].append(ann)
                ann_id += 1
    else:  # video
        cap = cv2.VideoCapture(input)
        last_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        pbar = tqdm.trange(int(last_frame))
        img_frames = []

        for img_id, frame in enumerate(pbar, start=1):
            ret, img = cap.read()
            h, w, _ = img.shape
            res = inference(model, img)
            if draw:
                img = draw.run(img, res)
                if cfg.save_imgs:
                    cv2.imwrite(
                        f"{cfg.output_dir}/imgs/frame_{frame}.jpg", img
                    )
                if cfg.save_video:
                    img_frames.append(img)
                if cfg.vis:
                    cv2.imshow("visualization", img)
                    cv2.waitKey(1)
            outputs["images"].append(
                {
                    "id": img_id,
                    "file_name": f"frame_{frame}",
                    "width": w,
                    "height": h,
                }
            )
            for ann in res:
                ann.update({"id": ann_id, "image_id": img_id})
                outputs["annotations"].append(ann)
                ann_id += 1

        if cfg.save_video:
            filename = f"{cfg.output_dir}/video/inference_result.mp4"
            video_out = cv2.VideoWriter(
                filename=filename,
                fourcc=cv2.VideoWriter_fourcc(*"DIVX"),
                fps=30,
                frameSize=(w, h),
            )
            for i in range(len(img_frames)):
                video_out.write(img_frames[i])
                print(f"write: {filename} | frame: {str(i)}")
            video_out.release()

    with open(f"{cfg.output_dir}/results.json", "w") as fp:
        json.dump(outputs, fp, indent=4, ensure_ascii=False)
    logger.info(f"Save inference results to {cfg.output_dir}/results.json")

    outputs = json.dumps(outputs, indent=4)
    return outputs


if __name__ == "__main__":
    args = make_parser().parse_args()
    run(**vars(args))