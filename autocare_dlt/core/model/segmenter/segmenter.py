from autocare_dlt.core.model.segmenter import BaseSegmenter


class Segmenter(BaseSegmenter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, **kwargs):
        features = self.backbone(x)
        features = self.neck(features)
        outputs = self.head(features, **kwargs)

        return outputs
