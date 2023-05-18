import torch
from sklearn.metrics import precision_recall_fscore_support


def cls_eval(logits, labels, training=False):
    sample_num = len(labels)
    _, preds = torch.max(logits, dim=1)

    preds = preds.view(sample_num, -1)
    labels = labels.view(sample_num, -1)

    metrics = {}
    metrics["accuracy"] = get_accuracy(preds, labels)
    if not training:
        precision, recall, f1 = get_precision_recall_f1(preds, labels)
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1

    return metrics


def get_accuracy(preds, labels):
    assert preds.shape == labels.shape

    n = preds.shape[0]

    accuracy = torch.sum(preds == labels) / n

    return accuracy.item()


def get_precision_recall_f1(preds, labels):
    assert preds.shape == labels.shape

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, beta=1, average="weighted", zero_division=1
    )

    return precision.item(), recall.item(), f1.item()


def multi_attr_eval(classes, outs, labels):
    res = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }
    for i, (k, v) in enumerate(classes.items()):
        preds = []
        for o in outs:
            preds.append(o[i])
        preds = torch.cat(preds).view(-1, len(v))
        lbs = labels[:, i].view(-1)
        sub_res = cls_eval(preds, lbs)
        for sk, sv in sub_res.items():
            res[sk].append(sv)
    return res
