import math
from functools import partial


class LRScheduler:
    def __init__(
        self, name, lr, iters_per_epoch, total_epochs, warmup=False, **kwargs
    ):
        """
        Supported lr schedulers: [cos, warmcos, multistep]

        Args:
            lr (float): learning rate.
            iters_per_peoch (int): number of iterations in one epoch.
            total_epochs (int): number of epochs in training.
            kwargs (dict):
                - cos: None
                - warmcos: [warmup_epochs, warmup_lr_start (default 1e-6)]
                - multistep: [milestones (epochs), gamma (default 0.1)]
        """
        self.name = name
        self.lr = lr
        self.iters_per_epoch = iters_per_epoch
        self.total_iters = iters_per_epoch * total_epochs
        self.warmup = warmup
        self.__dict__.update(kwargs)
        if warmup:
            if not hasattr(self, "warmup_epochs"):
                raise ValueError(
                    "you must set warmup_epochs value in lr_cfg when warmup=True"
                )
            if not (
                isinstance(self.warmup_epochs, int)
                or isinstance(self.warmup_epochs, float)
            ):
                raise ValueError(
                    "lr_cfg.warmup_epochs type must be int or float"
                )
            self.warmup_total_iters = int(
                self.iters_per_epoch * self.warmup_epochs
            )
            self.warmup_lr_start = getattr(self, "warmup_lr_start", 1e-6)
        if self.name is not None:
            self.lr_func = self._get_lr_func(name)

    def update_lr(self, iters):
        if self.warmup:
            lr = warm_lr(
                self.lr, self.warmup_total_iters, self.warmup_lr_start, iters
            )
            if self.name is not None:
                lr = (
                    self.lr_func(lr, iters)
                    if self.warmup_total_iters < iters
                    else lr
                )
            return lr
        else:
            return self.lr_func(self.lr, iters)

    def _get_lr_func(self, name):
        if name == "cosine":  # cosine lr schedule
            lr_func = partial(cos_lr, self.total_iters)
        elif name == "step":
            if not hasattr(self, "steps"):
                raise ValueError(
                    'you must set "steps" value in lr_cfg when type="step"'
                )
            if not (
                isinstance(self.steps, list) or isinstance(self.steps, tuple)
            ):
                raise ValueError("lr_cfg.steps type must be list or tuple")
            if not hasattr(self, "decay"):
                raise ValueError(
                    'you must set "decay" value in lr_cfg when type="step"'
                )
            if not isinstance(self.decay, float):
                raise ValueError("lr_cfg.decay type must be int or float")
            steps = [int(s * self.iters_per_epoch) for s in self.steps]
            decay = self.decay
            lr_func = partial(step_decay_lr, steps, decay)
        elif name == "linear":
            lr_func = partial(linear_decay_lr, self.total_iters)
        else:
            raise ValueError(f"Scheduler type {name} not supported.")
        return lr_func


def step_decay_lr(steps, decay, lr, iters):
    """step learning rate"""
    for step in steps:
        if iters >= step:
            lr *= decay
    return lr


def linear_decay_lr(total_iters, lr, iters):
    """Linear learning rate"""
    lr *= 1 - iters / total_iters
    return lr


def cos_lr(total_iters, lr, iters):
    """Cosine learning rate"""
    lr *= 0.5 * (1.0 + math.cos(math.pi * iters / total_iters))
    return lr


def warm_lr(lr, warmup_total_iters, warmup_lr_start, iters):
    """warm up."""
    if iters <= warmup_total_iters:
        lr = (lr - warmup_lr_start) * iters / float(
            warmup_total_iters
        ) + warmup_lr_start

    return lr
