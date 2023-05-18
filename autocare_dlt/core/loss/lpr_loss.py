import torch
from torch import nn
from torch.nn import functional as F


class LPRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        if torch.cuda.is_available():
            self.criterion = torch.nn.CrossEntropyLoss().cuda()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        batch_size = preds.size(0)

        preds = preds.permute(0, 2, 1)

        cost = self.criterion(preds, targets)
        return {"loss": cost}
