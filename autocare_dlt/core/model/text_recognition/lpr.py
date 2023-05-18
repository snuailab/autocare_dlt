import torch.nn as nn

from autocare_dlt.core.model.text_recognition import TextRecognition
from autocare_dlt.core.model.text_recognition.modules.bilstm import (
    BiLSTM,
    BiLSTM2,
)


class LicencePlateRecognition(TextRecognition):
    def __init__(self, model_cfg):
        super().__init__(model_cfg)

        """ Sequence modeling stage """
        if self.seq_cfg["name"] != "None":
            if self.seq_cfg["name"] == "BiLSTM":
                hidden_size = self.seq_cfg.get("hidden_size", 256)
                self.neck = LPRNeck(self.seq_cfg["input_size"], hidden_size)
                self.pred_cfg["input_size"] = hidden_size
        else:
            self.neck = None

        # self.trans = trans

    def forward(self, x, targets=None):

        x = self.backbone(x)
        """ Sequence modeling stage """
        if self.neck:
            x = self.neck(x)

        """ Prediction stage """
        x = self.head(x)

        # if x.size(1) < self.max_string_length:
        #     raise ValueError(f"w({x.size(1)}) of feature map is lower than max_string_length({self.max_string_length}), please increase img_size or decrease feature_index value")

        return x


class LPRNeck(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.seq = nn.Sequential(
            BiLSTM(input_size, hidden_size, hidden_size),
            BiLSTM2(hidden_size, hidden_size, hidden_size, seq_length=True),
        )

    def forward(self, x):
        return self.seq(x)
