from autocare_dlt.core.model.detector import BaseDetector


class SSD(BaseDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, **kwargs):
        img_size = x.shape[-2:]
        out_features = self.backbone(x)
        neck_outs = self.neck(out_features)

        outputs = self.head(neck_outs, img_size, **kwargs)

        return outputs
