import numpy as np
import torch
from torch import nn


# https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf
class ClassBalancedLoss(nn.Module):
    def __init__(self, beta: float = 0.9999, loss: str = "mse"):
        """Class Balanced Loss

        Args:
            beta (float, optional): beta. Defaults to 0.9999.
            loss_fn (str, optional): loss (mse, mae). Defaults to 'mse'.
        """

        self.weights = None
        self.beta = beta

        if loss == "mse":
            self.loss_fn = lambda a, b: torch.pow(a - b, 2)
        elif loss == "mae":
            self.loss_fn = lambda a, b: torch.abs(a - b)

    def register_count_dict(self, count_dict: dict):
        """register_count_dict must be called before training

        Args:
            count_dict (dict): _description_
        """

        self.weights = {}
        for property_name in count_dict:
            self.weights[property_name] = {}
            for value, count in count_dict[property_name]:
                en = (1 - np.power(self.beta, count)) / (1 - self.beta)
                en_inv = 1 / en
                self.weights[property_name][value] = en_inv

            en_inv_sum = sum(self.weights[property_name].values())
            for value, count in count_dict[property_name]:
                self.weights[property_name][value] /= en_inv_sum

        print("Class Balance Weights")
        print(self.weights)

    def forward(self, preds, labels):
        assert self.weights, BaseException("count_dict is not registered")

        se = self.loss_fn(preds, labels)
        se = (
            torch.tensor(
                self.weights[
                    labels.clone().cpu().squeeze().numpy().astype(int)
                ],
                device=preds.device,
            )
            * se
        )
        mse = torch.mean(se)

        return mse
