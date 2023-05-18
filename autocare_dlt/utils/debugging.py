import os

import cv2
import numpy as np
import torch

colors = [c for c in np.random.randint(0, 255, (1000, 3)).astype(int).tolist()]


def tensor2cv(tensor):

    image = tensor.cpu().numpy()
    image = image.transpose(1, 2, 0)
    image = (image - image.min()) / (image.max() - image.min()) * 255
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image


def plot_labels(
    input: torch.tensor, label_dict: dict, conf_thresh: float = 0.5
):

    image = tensor2cv(input)

    draw = image.copy()
    h, w = draw.shape[:2]

    labels = label_dict.get("labels", None)
    boxes = label_dict.get("boxes", None)
    scores = label_dict.get("scores", None)

    assert (labels is not None) and (boxes is not None)

    labels = labels.detach().cpu().numpy()
    boxes = boxes.detach().cpu().numpy().clip(0, 1)
    if scores is not None:
        scores = scores.detach().cpu().numpy()

    for i in range(len(labels)):
        label = labels[i]
        box = boxes[i]
        score = None if scores is None else scores[i]

        box[0::2] = box[0::2] * w
        box[1::2] = box[1::2] * h
        x1, y1, x2, y2 = box.astype(int)

        if score is None or score > conf_thresh:
            draw = cv2.rectangle(draw, (x1, y1), (x2, y2), colors[label], 2)

        # TODO
        # labels, scores
        if score is None:
            draw = cv2.putText(
                draw, f"{label}", (x1 + 5, y1 + 20), 1, 1.5, (0, 0, 0), 4
            )
            draw = cv2.putText(
                draw, f"{label}", (x1 + 5, y1 + 20), 1, 1.5, colors[label], 2
            )
        elif score > conf_thresh:
            draw = cv2.putText(
                draw,
                f"{label}_{score:.2f}",
                (x1 + 5, y1 + 20),
                1,
                1.5,
                (0, 0, 0),
                4,
            )
            draw = cv2.putText(
                draw,
                f"{label}_{score:.2f}",
                (x1 + 5, y1 + 20),
                1,
                1.5,
                colors[label],
                2,
            )

    return draw


def save_labels(
    input: torch.tensor,
    label: dict,
    pred: dict,
    save_path: str = None,
    prefix: int = 0,
):
    # print(image, label, pred, pseudo_label); exit()
    os.makedirs(save_path, exist_ok=True)

    ### labeled data
    # original image
    cv2.imwrite(
        os.path.join(save_path, f"{prefix}_input.png"), tensor2cv(input)
    )

    # label
    cv2.imwrite(
        os.path.join(save_path, f"{prefix}_label.png"),
        plot_labels(input, label),
    )

    # predction
    cv2.imwrite(
        os.path.join(save_path, f"{prefix}_prediction.png"),
        plot_labels(input, pred, conf_thresh=0.5),
    )
