from torch import nn


class Identity(nn.Identity):
    def __init__(self, *args, **kargs):
        super().__init__()
