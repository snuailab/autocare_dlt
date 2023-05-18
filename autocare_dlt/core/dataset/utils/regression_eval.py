import torch


def get_mae(preds, labels):
    return torch.mean(torch.abs(preds - labels)).item()


def get_mse(preds, labels):
    return torch.mean(torch.pow(preds - labels, 2)).item()


def get_rmse(preds, labels):
    return torch.sqrt(torch.mean(torch.pow(preds - labels, 2))).item()


def reg_eval(preds, labels, training=False):
    preds = preds.view(-1)
    labels = labels.view(-1)
    assert preds.shape == labels.shape

    metrics = {}
    metrics["mae"] = get_mae(preds, labels)
    metrics["mse"] = get_mse(preds, labels)
    metrics["rmse"] = get_rmse(preds, labels)

    return metrics
