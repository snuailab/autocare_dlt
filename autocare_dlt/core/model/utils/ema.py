from copy import deepcopy

import torch

from .functions import is_parallel


class ModelEMA:
    """
    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9996, max_iter=0):
        """
        Args:
            model (nn.Module): model to apply EMA.
            decay (float): ema decay reate.
            updates (int): counter of EMA updates.
        """
        # Create EMA(FP32)
        self.ema = deepcopy(
            model.module if is_parallel(model) else model
        ).eval()
        self.max_iter = max_iter
        self.decay = decay
        # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model, d=None, iter=-1):
        # Update EMA parameters
        with torch.no_grad():
            if d is None:
                d = self.get_decay(iter)

            msd = (
                model.module.state_dict()
                if is_parallel(model)
                else model.state_dict()
            )  # model state_dict
            for k, v in self.ema.state_dict().items():
                # if 'num_batches_tracked' in k: # EMAN update
                #     v = msd[k].detach()
                # else:
                # v *= d
                # v += (1.0 - d) * msd[k].detach()
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()

    def get_decay(self, i):
        if i < self.max_iter and i > 0:
            decay = 0.5 + (self.decay - 0.5) * (i - 1) / self.max_iter
        else:
            decay = self.decay
        return decay
