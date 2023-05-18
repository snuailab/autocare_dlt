import random

from autocare_dlt.core.model.pose_estimation import BasePoseEstimation


class PoseEstimation(BasePoseEstimation):
    def __init__(self, model_cfg):
        super().__init__(model_cfg)

    def forward(self, x, **kwargs):

        out_features = self.backbone(x)[0]
        neck_outs = self.neck(out_features)

        outputs = self.head(neck_outs)

        return outputs
