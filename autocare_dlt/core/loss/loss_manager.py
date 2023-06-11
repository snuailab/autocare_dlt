import sys
from typing import Tuple

import torch

from autocare_dlt.core.utils import SmartDict

from . import *


class DetLossManager:
    """Managing Multiple Loss functions"""

    def __init__(self, loss_cfg: dict, device: str = "cuda"):
        """initialize

        Args:
            loss_cfg (dict): loss config (in model_cfg.json)
        """

        self.loss_fns = []
        for loss_name, loss_info in loss_cfg.items():

            loss_type = loss_info["name"]
            loss_params = loss_info["params"]

            self.loss_fns.append(
                [
                    loss_name,
                    getattr(sys.modules[__name__], loss_type)(**loss_params),
                ]
            )

        self.device = device

    def __call__(self, outputs, targets) -> Tuple[torch.Tensor, SmartDict]:
        """calcuate losses

        Args:
            outputs (_type_): outputs of models
            targets (_type_): targets

        Returns:
            Tuple[torch.Tensor, SmartDict]: loss dictionary. {loss_name: loss_value}
        """

        res = SmartDict(
            default_value=lambda: torch.tensor(0.0, device=self.device)
        )

        for _, loss_fn in self.loss_fns:
            loss_dict = loss_fn(outputs, targets)
            res.add(loss_dict)
            break  # TODO: multiple loss

        return res.sum(), res

    def cuda(self, local_gpu_id=-1):
        self.device = (
            "cuda"
            if local_gpu_id == -1
            else torch.device(f"cuda:{local_gpu_id}")
        )


class ClsLossManager:
    """Managing Multiple Loss functions"""

    def __init__(self, loss_cfg: dict, device: str = "cuda"):
        """initialize

        Args:
            loss_cfg (dict): loss config (in model_cfg.json)
        """

        self.loss_fns = []
        for loss_name, loss_info in loss_cfg.items():

            loss_type = loss_info["name"]
            loss_params = loss_info["params"]

            self.loss_fns.append(
                [
                    loss_name,
                    getattr(sys.modules[__name__], loss_type)(**loss_params),
                ]
            )

        self.device = device

    def __call__(self, outputs, targets) -> Tuple[torch.Tensor, SmartDict]:
        """calcuate losses

        Args:
            outputs (_type_): outputs of models
            targets (_type_): targets

        Returns:
            Tuple[torch.Tensor, SmartDict]: loss dictionary. {loss_name: loss_value}
        """

        res = SmartDict(
            default_value=lambda: torch.tensor(0.0, device=self.device)
        )

        for loss_name, loss_fn in self.loss_fns:

            for output, target in zip(outputs, targets):
                res[loss_name] += loss_fn(output, target)

        return res.sum(), res

    def cuda(self):
        self.device = "cuda"


class STRLossManager:
    def __init__(self, loss_cfg: dict, device: str = "cuda") -> None:
        """initialize

        Args:
            loss_cfg (dict): loss config (in model_cfg.json)
        """

        self.loss_fns = []
        for loss_name, loss_info in loss_cfg.items():

            loss_type = loss_info["name"]
            loss_params = loss_info["params"]

            self.loss_fns.append(
                [
                    loss_name,
                    getattr(sys.modules[__name__], loss_type)(**loss_params),
                ]
            )

        self.device = device

    def __call__(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, SmartDict]:
        """Calculate losses

        TODO: Description.

        Args:
            outputs (torch.Tensor): _description_
            targets (torch.Tensor): _description_
            target_weights (torch.Tensor): _description_

        Returns:
            Tuple[torch.Tensor, SmartDict]: _description_
        """
        res = SmartDict(
            default_value=lambda: torch.tensor(0.0, device=self.device)
        )

        for _, loss_fn in self.loss_fns:
            loss_dict = loss_fn(outputs, targets)
            res.add(loss_dict)
            break  # TODO: multiple loss

        return torch.Tensor(res.sum()), res

    def cuda(self) -> None:
        self.device = "cuda"


class PoseLossManager:
    def __init__(self, loss_cfg: dict, device: str = "cuda") -> None:
        """initialize

        Args:
            loss_cfg (dict): loss config (in model_cfg.json)
        """

        self.loss_fns = []
        for loss_name, loss_info in loss_cfg.items():

            loss_type = loss_info["name"]
            loss_params = loss_info["params"]

            self.loss_fns.append(
                [
                    loss_name,
                    getattr(sys.modules[__name__], loss_type)(**loss_params),
                ]
            )

        self.device = device

    def __call__(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, SmartDict]:
        """Calculate losses

        TODO: Description.

        Args:
            outputs (torch.Tensor): _description_
            targets (torch.Tensor): _description_
            target_weights (torch.Tensor): _description_

        Returns:
            Tuple[torch.Tensor, SmartDict]: _description_
        """
        res = SmartDict(
            default_value=lambda: torch.tensor(0.0, device=self.device)
        )

        for _, loss_fn in self.loss_fns:
            loss_dict = loss_fn(outputs, targets)
            res.add(loss_dict)
            break  # TODO: multiple loss

        return torch.Tensor(res.sum()), res

    def cuda(self) -> None:
        self.device = "cuda"


class SegLossManager:
    def __init__(self, loss_cfg: dict, classes: list, device: str = "cuda"):
        """initialize

        Args:
            loss_cfg (dict): loss config (in model_cfg.json)
        """
        
        self.loss_fns = []
        for loss_name, loss_info in loss_cfg.items():

            loss_type = loss_info["name"]
            loss_params = loss_info["params"]
            loss_params["classes"] = classes

            self.loss_fns.append(
                [
                    loss_name,
                    getattr(sys.modules[__name__], loss_type)(**loss_params),
                ]
            )

        self.device = device

    def __call__(self, outputs, targets) -> Tuple[torch.Tensor, SmartDict]:
        """calcuate losses

        Args:
            outputs (_type_): outputs of models
            targets (_type_): targets

        Returns:
            Tuple[torch.Tensor, SmartDict]: loss dictionary. {loss_name: loss_value}
        """

        res = SmartDict(
            default_value=lambda: torch.tensor(0.0, device=self.device)
        )

        for loss_name, loss_fn in self.loss_fns:

            for output, target in zip(outputs, targets):
                loss_dict = loss_fn(output, target)
                res.add(loss_dict)

        return res.sum(), res

    def cuda(self):
        self.device = "cuda"