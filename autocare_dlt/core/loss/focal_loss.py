import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


class BCE_FocalLoss(nn.Module):
    def __init__(
        self,
        alpha=None,
        gamma=1.5,
        reduction="mean",
        pos_weight=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        if not gamma >= 0:
            raise ValueError(f"gamma :{gamma} must be larger than 0")
        if alpha is None:
            alpha = -1.0
        if alpha > 1.0:
            raise ValueError(
                f"alpha :{alpha} must be float value smaller than 1 (alpha is off when alpha < 0 or alpha=None)"
            )
        if reduction not in ["mean", "sum", "none"]:
            raise KeyError(f"reduction: {reduction} is not supported.")
        self.alpha = alpha
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        if torch.cuda.is_available():
            self.bce = nn.BCEWithLogitsLoss(
                pos_weight=pos_weight, reduction="none"
            ).cuda()
        else:
            self.bce = nn.BCEWithLogitsLoss(
                pos_weight=pos_weight, reduction="none"
            )

    def forward(self, inputs, targets):
        # focal loss
        if len(targets.shape) == 1 or targets.shape[1] == 1:  # (N)
            targets = F.one_hot(targets.view(-1), inputs.shape[1])
            targets = targets.type(inputs.dtype)
        BCE = self.bce(inputs, targets)
        p = torch.exp(-BCE)
        loss = (1 - p) ** self.gamma * BCE

        if self.alpha > 0:
            alpha_weight = targets * self.alpha + (1 - targets) * (
                1 - self.alpha
            )
            loss *= alpha_weight
        loss = loss.sum(-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class CE_FocalLoss(nn.Module):
    def __init__(
        self, gamma=0, size_average=True, ignore_index=-1, *args, **kwargs
    ):
        super().__init__()
        if not gamma >= 0:
            raise ValueError(f"gamma :{gamma} must be larger than 0")
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, input, target):

        train_index = target != self.ignore_index
        target = target[train_index]
        input = input[train_index.view(-1)]
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
