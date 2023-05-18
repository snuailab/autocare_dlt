from autocare_dlt.core.model.classifier import BaseClassifier


class Classifier(BaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, **kwargs):
        features = self.backbone(x)
        features = self.neck(features)
        outputs = self.head(features, **kwargs)

        return outputs
