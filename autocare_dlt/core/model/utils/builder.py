import sys

import torch

from autocare_dlt.core.model.classifier import *
from autocare_dlt.core.model.detector import *
from autocare_dlt.core.model.pose_estimation import *
from autocare_dlt.core.model.regressor import *
from autocare_dlt.core.model.text_recognition import *
from autocare_dlt.core.utils.checkpoint import load_ckpt
from autocare_dlt.utils.config import *


def build_model(cfg, strict=False):
    classes = cfg["classes"]
    model = getattr(sys.modules[__name__], cfg["task"])(model_cfg=cfg["model"])
    ckpt = cfg.ckpt

    if isinstance(ckpt, str):
        ckpt_dict = torch.load(ckpt, map_location="cpu")
        eval_ckpt = (
            ckpt_dict["model"]
            if ckpt_dict.get("model_ema", None) is None
            else ckpt_dict["model_ema"]
        )
        model = load_ckpt(model, eval_ckpt, strict=strict)
    elif isinstance(ckpt, dict):
        eval_ckpt = ckpt["model"] if ckpt.get("model", False) else ckpt
        model = load_ckpt(model, eval_ckpt, strict=strict)
    elif ckpt == None:
        pass
    else:
        raise TypeError(f"ckpt type ({type(cfg.ckpt)}) is not supported")

    return model, classes
