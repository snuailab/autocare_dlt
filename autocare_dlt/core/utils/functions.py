import torch
from loguru import logger
from torchinfo import summary


def check_gpu_availability(
    model,
    input_size: list,
    batch_size: int,
    dtype: torch.dtype,
    gpu_total_mem: int,
) -> bool:
    """Return error log when expected GPU out of memory

    Args:
        model: Model
        input_size (list): Input size of model.
        batch_size (int): Batch size.
        dtype (torch.dtype): Data type of tensor.
        gpu_total_mem (int): GPU memory being currently used.
    """
    input_size = input_size * 2 if len(input_size) == 1 else input_size
    model_summary = summary(
        model=model,
        input_size=(1, 3, input_size[1], input_size[0]),
        dtypes=[dtype],
        verbose=0,
    )

    model_total_mem = (
        model_summary.total_input
        + model_summary.total_output_bytes
        + model_summary.total_param_bytes
    ) * batch_size  # Estimated model size. (Bytes)
    # Formula referenced from
    # ``/site-packages/torchinfo/model_statistics.py``

    model_total_mem *= 1.1  # Margin

    model_total_mem = float(model_total_mem) / (1024.0**2)  # MB
    gpu_total_mem = float(gpu_total_mem) / (1024.0**2)  # MB

    # TODO: loggers will be moved to base trainer
    availability = True
    if model_total_mem <= gpu_total_mem:
        logger.info(f"Model summary:\n {model_summary}")
        logger.info(f"Model total memory: {model_total_mem:.2f} MB")
        logger.info(f"GPU total memory: {gpu_total_mem:.2f} MB")

    elif model_total_mem > gpu_total_mem:
        availability = False
        logger.info(f"Model summary:\n {model_summary}")
        logger.critical(
            "Expected GPU out of memory (Model size {:.2f} MB > GPU total memory {:.2f} MB). Consider reducing the batch size.".format(
                model_total_mem, gpu_total_mem
            )
        )

    return availability


def det_labels_to_cuda(labels, gpu_id=-1):
    for l in labels:
        if l["labels"] is not None:
            l["labels"] = (
                l["labels"].cuda()
                if gpu_id == -1
                else l["labels"].cuda(gpu_id)
            )
            l["boxes"] = (
                l["boxes"].cuda() if gpu_id == -1 else l["boxes"].cuda(gpu_id)
            )


def key_labels_to_cuda(labels):
    for l in labels:
        if l["heatmap"] is not None:
            l["heatmap"] = l["heatmap"].cuda()
            # l['joints'] = l['joints'].cuda()


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=None):
        if isinstance(val, int) or isinstance(val, float):
            n = 1
        else:
            if n is None:
                n = len(val) if len(val.shape) > 0 else 1
            val = val.mean().item()

        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count != 0 else 0
