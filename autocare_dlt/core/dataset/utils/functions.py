import cv2
import numpy as np
import torch


def read_img_rect(path, img_size):
    if "\n" in path:
        path = path[:-1]
    img = cv2.imread(path)  # BGR
    if img is None:
        raise Exception("Image Not Found " + path)
    h0, w0 = img.shape[:2]  # orig hw
    r = img_size / max(h0, w0)
    img = cv2.resize(
        img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_AREA
    )
    return img, h0, w0


def read_img(path, img_size: list):
    if "\n" in path:
        path = path[:-1]
    img = cv2.imread(path)  # BGR
    if img is None:
        raise Exception("Image Not Found " + path)
    h0, w0 = img.shape[:2]  # orig hw
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
    return img, h0, w0


def collate_fn(batch):
    if len(batch[0]) == 3:
        imgs, imgs_s, labels = zip(*batch)  # transposed
        return (torch.stack(imgs, 0), torch.stack(imgs_s, 0)), labels
    else:
        imgs, labels = zip(*batch)  # transposed
        return torch.stack(imgs, 0), labels


def letterbox(
    im,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (
        new_shape[1] - new_unpad[0],
        new_shape[0] - new_unpad[1],
    )  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = (
            new_shape[1] / shape[1],
            new_shape[0] / shape[0],
        )  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, ratio, (dw, dh)


class DataIterator:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data_iter = iter(self.data_loader)

    def __call__(self):
        try:
            outs = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            outs = next(self.data_iter)
        return outs


def img2tensor(img):
    if not torch.is_tensor(img):
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img)
    return img
