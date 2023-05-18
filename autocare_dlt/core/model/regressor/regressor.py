from autocare_dlt.core.model.regressor.base_regressor import BaseRegressor


class Regressor(BaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        features = self.backbone(x)
        features = self.neck(features)
        outputs = self.head(features)

        return outputs
