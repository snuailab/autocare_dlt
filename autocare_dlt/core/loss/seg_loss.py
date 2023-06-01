import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class SegLoss(nn.Module):
    def __init__(self, class_weights, classes):
        super().__init__()
        weights = torch.ones(len(classes))
        indices = self.find_indices(list(class_weights.keys()), classes)
        weights[indices] = torch.FloatTensor(list(class_weights.values()))
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        if torch.cuda.is_available():
            self.criterion = self.criterion.cuda()
    
    def forward(self, preds, targets):
        targets = targets["labels"]
        preds = preds.unsqueeze(dim=0)
        targets = targets.unsqueeze(dim=0).long()
        loss = self.criterion(preds, targets)

        return {"loss": loss}

    def find_indices(self, a, b):
        indices = []
        for item in a:
            if item in b:
                indices.append(b.index(item))
            else:
                indices.append(None)
        return indices