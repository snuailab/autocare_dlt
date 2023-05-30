import argparse
import os
import sys
from typing import Union

from box import Box

sys.path.append(os.getcwd())
from loguru import logger

from autocare_dlt.core.model import build_model
from autocare_dlt.utils.config import *


def make_parser():
    parser = argparse.ArgumentParser("Tx onnx deploy")
    parser.add_argument(
        "--output_name",
        type=str,
        default="model",
        help="output name of models",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model_cfg",
        type=str,
        help="model configuration file",
        required=True,
    )
    parser.add_argument(
        "--input_size",
        nargs="+",
        default=[],
        type=int,
        help="input size of model",
    )
    parser.add_argument(
        "-o", "--opset", default=11, type=int, help="onnx opset version"
    )
    parser.add_argument(
        "--no-onnxsim", action="store_true", help="use onnxsim or not"
    )
    parser.add_argument(
        "-c", "--ckpt", default=None, type=str, help="ckpt path"
    )

    return parser


def run(
    output_name: str,
    model_cfg: str,
    ckpt: Union[str, dict],
    input_size: list = None,
    opset: int = 11,
    no_onnxsim: bool = False,
) -> None:
    """Export onnx file

    Args:
        output_name (str): file name for onnx output (.onnx)
        model_cfg (str): path for model configuration file
        ckpt (Union[str, dict]): path for checkpoint file or state dict
        input_size (list, optional): input size of model. use model config value if input_size is None. Default to None.
        opset (int, optional): onnx opset version. Defaults to 11.
        no_onnxsim (bool, optional): whether to use onnxsim or not. Defaults to False.
    """
    import torch

    args = Box(
        {
            "output_name": output_name,
            "model_cfg": model_cfg,
            "input_size": input_size,
            "ckpt": ckpt,
            "opset": opset,
            "no_onnxsim": no_onnxsim,
        }
    )

    cfg = parsing_config(args)

    logger.info(f"args value: {args}")
    output_name = args.output_name
    if not output_name.endswith(".onnx"):
        output_name = output_name + ".onnx"

    logger.info("Build Model..")
    model, _ = build_model(cfg, strict=True)
    model.eval()

    logger.info("loading checkpoint done.")
    input_name = ["inputs"]

    input_size = cfg.input_size
    if len(input_size) == 1:
        input_size = [input_size[0], input_size[0]]
        logger.warning(
            f"input size is single int value(={input_size[0]}), please check your inferece stream resolution."
        )
    elif len(input_size) == 2:
        pass
    else:
        raise BaseException(f"Invalid input_size : {input_size}")

    dynamic_batch = 16 if torch.cuda.is_available() else 1
    if cfg["task"] in detector_list:
        outputs_list = ["bbox", "conf", "class_id"]
        dummy_input = torch.randn(
            dynamic_batch, 3, input_size[1], input_size[0]
        )
    elif cfg["task"] in classifier_list + str_list:
        outputs_list = ["predictions"]
        dummy_input = torch.randn(
            dynamic_batch, 3, input_size[1], input_size[0]
        )
    elif cfg["task"] in pose_estimator_list:
        outputs_list = ["keypoints"]
        dummy_input = torch.randn(
            dynamic_batch, 3, input_size[1], input_size[0]
        )

    torch.onnx.export(
        model,
        dummy_input,
        output_name,
        input_names=input_name,
        output_names=outputs_list,
        opset_version=args.opset,
        dynamic_axes={
            name: {0: "batch_size"} for name in input_name + outputs_list
        },
    )
    logger.info(f"generated onnx model named {output_name}")

    if not args.no_onnxsim:
        import onnx
        from onnxsim import simplify

        # use onnxsimplify to reduce reduent model.
        onnx_model = onnx.load(output_name)
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, output_name)
        logger.info(f"generated simplified onnx model named {output_name}")


if __name__ == "__main__":
    args = make_parser().parse_args()
    run(**vars(args))
