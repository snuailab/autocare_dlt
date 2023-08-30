from torch import nn


class SegmentationHead(nn.Module):
    def __init__(self, num_classes, num_cls_per_attributes=None, model_size="L"):
        super().__init__()
        self.num_classes = num_classes
        self.num_cls_per_attributes = (
            num_cls_per_attributes if num_cls_per_attributes else [num_classes]
        )

    def forward(self, logits):
        if self.training:
            out = logits
        else:
            out = self.post_processing(logits)

        return out

    def post_processing(self, logits):
        preds = []
        for att_logits in logits:
            att_logits = att_logits
            preds.append(att_logits.cpu().detach())

        return preds
