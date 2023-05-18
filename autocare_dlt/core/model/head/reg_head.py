from torch import nn


class RegressionHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_cls_per_attributes=None):
        super().__init__()

        self.num_classes = num_classes
        self.num_cls_per_attributes = (
            num_cls_per_attributes if num_cls_per_attributes else [num_classes]
        )

        self.head = nn.ModuleList()
        for ncpa in self.num_cls_per_attributes:
            self.head.append(nn.Linear(in_channels, ncpa))

    def forward(self, features):
        logits = []
        for h in self.head:
            logits.append(h(features))

        if self.training:
            out = logits
        else:
            out = self.post_processing(logits)

        return out

    def post_processing(self, logits):
        preds = []
        for att_logits in logits:
            att_logits = att_logits.unsqueeze(2).unsqueeze(2)
            preds.append(att_logits.cpu().detach())

        return preds
