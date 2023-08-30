from autocare_dlt.core.model.segmenter import BaseSegmenter


class Segmenter(BaseSegmenter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, **kwargs):
        # TODO: just for UNet custom. have to change for normal cases
        logit, features = self.backbone(x)
        features = self.neck(features)
        outputs = self.head(logit, **kwargs)

        return outputs
