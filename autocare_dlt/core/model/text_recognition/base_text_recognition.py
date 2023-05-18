import timm
import torch
import torch.nn as nn

from autocare_dlt.core.model.text_recognition.modules.attention import (
    Attention,
)
from autocare_dlt.core.model.text_recognition.modules.bilstm import BiLSTM


class TextRecognition(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        self.model_cfg = model_cfg
        self.max_string_length = model_cfg.get("max_string_length", 10)

        self.trans_cfg = model_cfg.pop("Transformation", None)
        self.feat_cfg = model_cfg.pop("FeatureExtraction", None)
        self.seq_cfg = model_cfg.pop("SequenceModeling", None)
        self.pred_cfg = model_cfg.pop("Prediction", None)

        # """ Transformation stage """ #TODO: Need it?
        # if trans_cfg["name"] != "None":
        #     if trans_cfg["name"] == "TPS":
        #         num_fiducial = trans_cfg.get("num_fiducial", 20)
        #         img_size = trans_cfg["img_size"]

        #         trans = TPS(
        #             F=num_fiducial,
        #             I_size=(img_size[1], img_size[0]),
        #             I_r_size=(img_size[1], img_size[0]),
        #             I_channel_num=3 #TODO: add channel in cfg
        #         )
        # else:
        #     trans = None

        """ Feature extraction stage """
        backbone_name = self.feat_cfg.pop("name", None)

        self.backbone = TextRecognitionBackbone(
            backbone_name,
            self.feat_cfg.get("feature_index", -1),
            self.feat_cfg["output_size"],
        )

        """ Sequence modeling stage """
        if self.seq_cfg["name"] != "None":
            if self.seq_cfg["name"] == "BiLSTM":
                hidden_size = self.seq_cfg.get("hidden_size", 256)
                self.neck = TextRecognitionNeck(
                    self.seq_cfg["input_size"], hidden_size
                )
                self.pred_cfg["input_size"] = hidden_size
        else:
            self.neck = None

        """ Prediction stage """
        if self.pred_cfg["name"] == "CTC":
            self.head = TextRecognitionHead(
                self.pred_cfg["input_size"], self.pred_cfg["num_classes"]
            )
        elif self.pred_cfg["name"] in ["Attention", "Attn"]:
            # 현재는 작동되지 않음
            # TODO: Attn 구현
            hidden_size = self.pred_cfg.get("hidden_size", 256)
            self.head = Attention(
                self.pred_cfg["input_size"],
                hidden_size,
                self.pred_cfg["num_classes"],
            )
        else:
            raise ValueError("Prediction is one of [CTC, Attn]")

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


class TextRecognitionBackbone(nn.Module):
    def __init__(self, backbone_name, feature_index, output_size):
        super().__init__()
        self.feat = timm.create_model(
            backbone_name, features_only=True, pretrained=True
        )
        self.feature_index = feature_index
        self.adaptive_pool = nn.AdaptiveAvgPool2d((output_size, 1))

    def forward(self, x):
        x = self.feat(x)[self.feature_index]
        x = x.permute(0, 3, 1, 2)
        x = self.adaptive_pool(x).squeeze(3)  # [b, c, h, w] -> [b, w, c]
        return x


class TextRecognitionNeck(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.seq = nn.Sequential(
            BiLSTM(input_size, hidden_size, hidden_size),
            BiLSTM(hidden_size, hidden_size, hidden_size),
        )

    def forward(self, x):
        return self.seq(x)


class TextRecognitionHead(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.pred = nn.Linear(input_size, num_classes)

    def forward(self, x, **kwargs):
        logits = self.pred(x.contiguous())
        if self.training:
            out = logits
        else:
            out = self.post_processing(logits)

        return out

    def post_processing(self, logits):

        out = torch.softmax(logits, -1)

        return out.cpu().detach()
