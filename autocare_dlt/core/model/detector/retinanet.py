import random

from autocare_dlt.core.model.detector import BaseDetector


class RetinaNet(BaseDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, **kwargs):
        img_size = x.shape[-2:]
        use_pose_mum = kwargs.get("use_pose_mum", False)
        out_features = self.backbone(x)
        neck_outs = self.neck(out_features)
        outputs = self.head(neck_outs, img_size, **kwargs)

        return outputs
