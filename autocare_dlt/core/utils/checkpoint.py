import os
import re

import torch
from loguru import logger


def load_ckpt(
    model,
    ckpt,
    remove_prefix=None,
    add_prefix=None,
    keys_to_include=None,
    add_particular_prefix=None,
    keys_to_ignore=None,
    strict=False,
):
    model_state_dict = model.state_dict()
    ckpt = edit_keys(
        ckpt,
        remove_prefix=remove_prefix,
        add_particular_prefix=add_particular_prefix,
        add_prefix=add_prefix,
        keys_to_ignore=keys_to_ignore,
        keys_to_include=keys_to_include,
    )
    # load multiGPU trained ckpts
    if not ("module" in list(model_state_dict.keys())[0]) and (
        "module" in list(ckpt.keys())[0]
    ):
        non_ddp_dict = {}
        for k, v in ckpt.items():
            non_ddp_dict[k[7:]] = v
        ckpt = non_ddp_dict
    # for distirbuted mode - load singleGPU ckpt to DDP model
    if ("module" in list(model_state_dict.keys())[0]) and not (
        "module" in list(ckpt.keys())[0]
    ):
        ddp_dict = {}
        for k, v in ckpt.items():
            k = "module." + k
            ddp_dict[k] = v
        ckpt = ddp_dict

    load_dict = {}
    for key_model, v in model_state_dict.items():
        if key_model not in ckpt:
            logger.warning(
                "{} is not in the ckpt. Please double check and see if this is desired.".format(
                    key_model
                )
            )
            continue
        v_ckpt = ckpt[key_model]
        if v.shape != v_ckpt.shape:
            if strict:
                logger.error(
                    "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                        key_model, v_ckpt.shape, key_model, v.shape
                    )
                )
            else:
                logger.warning(
                    "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                        key_model, v_ckpt.shape, key_model, v.shape
                    )
                )
                continue
        load_dict[key_model] = v_ckpt

    model.load_state_dict(load_dict, strict=strict)
    return model


def save_checkpoint(state, is_best, save_dir, model_name=""):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, model_name + "_ckpt.pth")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save_dir, "best_ckpt.pth")
        torch.save(
            {
                "model": state["model_ema"]
                if state.get("model_ema", None) is not None
                else state["model"]
            },
            best_filename,
        )


def edit_keys(
    d,
    remove_prefix=None,
    add_particular_prefix=None,
    add_prefix=None,
    keys_to_include=None,
    keys_to_ignore=None,
):
    """
    Remove (only if possible) and then add the respective prefix for all keys in the dict d, retain keys, matching a
    regular expression given in `keys_to_include` (if not None), remove keys matching a regular expression in
    `keys_to_ignore` (if not None), and return the result.
    """

    if type(keys_to_ignore) is str and len(keys_to_ignore) > 0:
        keys_to_ignore = [keys_to_ignore]
    if type(keys_to_include) is str and len(keys_to_include) > 0:
        keys_to_include = [keys_to_include]
    keys_to_ignore = keys_to_ignore or ["^$"]
    keys_to_include = keys_to_include or [".*"]

    def should_add(key):
        should_include = True in [
            (re.match(pattern, key) is not None) for pattern in keys_to_include
        ]
        should_ignore = True in [
            (re.match(pattern, key) is not None) for pattern in keys_to_ignore
        ]
        return should_include and not should_ignore

    if remove_prefix is not None:
        length = len(remove_prefix)
        d = {
            k[length:] if k.startswith(remove_prefix) else k: v
            for k, v in d.items()
        }
    if add_particular_prefix is not None:
        d = {
            add_particular_prefix + k
            if k.startswith(add_particular_prefix)
            else k: v
            for k, v in d.items()
        }
    if add_prefix is not None:
        d = {add_prefix + k: v for k, v in d.items() if add_prefix in k}

    return {k: v for k, v in d.items() if should_add(k)}
