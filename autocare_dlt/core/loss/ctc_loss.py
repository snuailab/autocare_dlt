import torch
from torch import nn
from torch.nn import functional as F


class STRCTCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        if torch.cuda.is_available():
            self.criterion = torch.nn.CTCLoss(zero_infinity=True).cuda()
        else:
            self.criterion = torch.nn.CTCLoss(zero_infinity=True)

    def forward(self, preds, targets):
        batch_size = preds.size(0)

        input_lengths = torch.IntTensor([preds.size(1)] * batch_size)
        target_lengths = torch.IntTensor([targets.size(1)] * batch_size)
        if torch.cuda.is_available():
            input_lengths = input_lengths.cuda()
            target_lengths = target_lengths.cuda()

        preds = preds.log_softmax(2).permute(1, 0, 2)  # to use CTCLoss format

        torch.backends.cudnn.enabled = False
        cost = self.criterion(
            preds, targets.reshape(-1), input_lengths, target_lengths
        )
        torch.backends.cudnn.enabled = True
        return {"loss": cost}
