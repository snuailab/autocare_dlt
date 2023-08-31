import copy
import json
import os
from collections import OrderedDict

from box import Box

detector_list = ["SSD", "RetinaNet", "YOLOv5", "SSD4Point"]
classifier_list = ["Classifier"]
regressor_list = ["Regressor"]
str_list = ["TextRecognition", "LicencePlateRecognition"]
pose_estimator_list = ["PoseEstimation"]
segmenter_list = ["Segmenter"]

def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise TypeError


def json_to_dict(json_path):
    if json_path is None:
        return {}
    with open(json_path) as json_file:
        return json.load(json_file)


def parsing_config(args: Box) -> Box:
    cfg = OrderedDict()
    model = json_to_dict(args.pop("model_cfg", None))
    data = json_to_dict(args.pop("data_cfg", None))

    cfg.update(model)
    cfg.update(data)
    # NOTE: this puts the terminal input args into `cfg`
    for k, v in args.items():
        if v is not None:
            cfg[k] = v
        else:
            if k not in cfg:
                cfg[k] = v

    ema_cfg = (
        Box(
            {
                "type": "ema",
                "decay": 0.9996,
                "burn_in_epoch": 0,
                **cfg.get("ema_cfg", {}),
            }
        )
        if cfg.get("ema")
        else None
    )
    cfg.update({"ema_cfg": ema_cfg})

    # TODO: include the cfgs into `cfg` that's required for HPE

    if cfg.get("data", False):
        cfg["data"]["task"] = cfg.get("task", None)
        cfg["data"]["classes"] = cfg.get("classes", ["object"])
        cfg["data"]["num_classes"] = cfg.get("num_classes", 1)

    if cfg["task"] in str_list:
        cfg["model"]["Prediction"]["num_classes"] = (
            cfg["num_classes"] + 1
        )  # add blank
        if cfg.get("data", None):  # train
            cfg["model"]["Transformation"]["img_size"] = cfg["data"][
                "img_size"
            ]
            cfg["data"]["max_string_length"] = cfg["model"][
                "max_string_length"
            ]
        else:  # inference
            cfg["model"]["Transformation"]["img_size"] = cfg["input_size"]
    else:
        cfg["model"]["head"]["num_classes"] = cfg["num_classes"]

    if cfg["task"] == "YOLOv5":  # TODO
        for _, loss_cfg in cfg["loss"].items():
            if loss_cfg["name"] == "YoloLoss":
                loss_cfg["params"]["num_classes"] = len(cfg["classes"])
                loss_cfg["params"]["anchors"] = cfg["model"]["head"].get(
                    "anchors", None
                )

    if cfg.get("output_dir", False) == False:
        return Box(cfg)
    
    if cfg.get("exp_name", False):
        output_path = os.path.join(cfg["output_dir"], cfg["exp_name"])
    else:
        output_path = cfg["output_dir"]

    cfg["output_path"] = output_path
    if (
        not cfg.get("overwrite", False)
        and not cfg.get("resume", False)
        and os.path.exists(output_path)
    ):
        if cfg.get("exp_name", False):    
            exp_name = cfg["exp_name"]
            raise KeyError(f"exp_name {exp_name} is already exist")
        else:
            output_dir = cfg["output_dir"]
            raise KeyError(f"output directory {output_dir} is already exist")
        
    # TODO: multi head config
    # cls_per_attr = [len(a) for attr, a in self.classes.items()]
    # self.cfg["model"]["head"]["num_cls_per_attributes"] = cls_per_attr
    # if sum(cls_per_attr) != self.cfg["num_classes"]:
    #     raise BaseException(
    #         "[num_classes] is not same with sum of [num_cls_per_attributes]"
    #     )

    return Box(cfg)


def save_cfg(output_path, cfg):
    cfg_ = {
        **cfg,
        "ckpt": cfg.get("ckpt", None)
        if isinstance(cfg.get("ckpt", None), str)
        else None,
    }
    os.makedirs(output_path, exist_ok=True)
    cfg_path = output_path + "/config.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg_, f, ensure_ascii=False, indent=4)
