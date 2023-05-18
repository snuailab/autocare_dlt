import math

import numpy as np
import torch
import torchvision


def nms(results, detections_per_img, nms_thresh, min_score=0.05):
    detections = []
    for batch_res in results:
        batch_boxes, batch_scores, batch_labels = batch_res
        for image_boxes, image_scores, image_labels in zip(
            batch_boxes, batch_scores, batch_labels
        ):
            if image_boxes[0].shape[0] == 8:
                image_boxes_2point = image_boxes[:, [0, 1, 4, 5]]
                index = image_scores > min_score
                keep = torchvision.ops.batched_nms(
                    image_boxes_2point[index],
                    image_scores[index],
                    image_labels[index],
                    nms_thresh,
                )
                keep = keep[:detections_per_img]
            else:
                index = image_scores > min_score
                keep = torchvision.ops.batched_nms(
                    image_boxes[index],
                    image_scores[index],
                    image_labels[index],
                    nms_thresh,
                )
                keep = keep[:detections_per_img]
            detections.append(
                {
                    "boxes": image_boxes[index][keep],
                    "scores": image_scores[index][keep],
                    "labels": image_labels[index][keep],
                }
            )
    return detections


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_b.shape[1] != 4 or (
        bboxes_a.shape[1] != 4 and bboxes_a.shape[1] != 8
    ):
        raise IndexError
    if bboxes_a.shape[1] == 8:
        if xyxy:
            tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
            br = torch.min(bboxes_a[:, None, 4:6], bboxes_b[:, 2:])
            area_a = torch.prod(bboxes_a[:, 4:6] - bboxes_a[:, :2], 1)
            area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
        else:
            raise ValueError("Not implemented yet")

        en = (tl < br).type(tl.type()).prod(dim=2)
        area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
        return area_i / (area_a[:, None] + area_b - area_i)
    else:
        if xyxy:
            tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
            br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
            area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
            area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
        else:
            tl = torch.max(
                (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
            )
            br = torch.min(
                (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
            )

            area_a = torch.prod(bboxes_a[:, 2:], 1)
            area_b = torch.prod(bboxes_b[:, 2:], 1)
        en = (tl < br).type(tl.type()).prod(dim=2)
        area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
        return area_i / (area_a[:, None] + area_b - area_i)


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def xyxy2xywh(bboxes):
    return torch.hstack((bboxes[:, 0:2], bboxes[:, 2:4] - bboxes[:, 0:2]))


def xyxy2cxcywh(bboxes):
    new_bboxes = bboxes.clone()
    new_bboxes[:, 2] = new_bboxes[:, 2] - new_bboxes[:, 0]
    new_bboxes[:, 3] = new_bboxes[:, 3] - new_bboxes[:, 1]
    new_bboxes[:, 0] = new_bboxes[:, 0] + new_bboxes[:, 2] * 0.5
    new_bboxes[:, 1] = new_bboxes[:, 1] + new_bboxes[:, 3] * 0.5
    return new_bboxes


def cxcywh2xywh(bboxes):
    new_bboxes = bboxes.clone()
    new_bboxes[:, 0] = new_bboxes[:, 0] - new_bboxes[:, 2] * 0.5
    new_bboxes[:, 1] = new_bboxes[:, 1] - new_bboxes[:, 3] * 0.5
    return new_bboxes


def cxcywh2xyxy(bboxes, batch_input=False):
    if batch_input:
        x1 = bboxes[:, :, 0] - bboxes[:, :, 2] * 0.5
        y1 = bboxes[:, :, 1] - bboxes[:, :, 3] * 0.5
        x2 = bboxes[:, :, 2] + x1
        y2 = bboxes[:, :, 3] + y1
    else:
        x1 = bboxes[:, 0] - bboxes[:, 2] * 0.5
        y1 = bboxes[:, 1] - bboxes[:, 3] * 0.5
        x2 = bboxes[:, 2] + x1
        y2 = bboxes[:, 3] + y1
    return torch.stack((x1, y1, x2, y2), -1)


def prediction2pseudolabel(predictions, input_size):
    output = torch.zeros((len(predictions), 50, 5))  # (bs,50,5)
    for i, pred in enumerate(predictions):
        if pred is not None:
            pred[:, :4] = pred[:, :4].clamp(min=0, max=input_size[0])
            bbox = xyxy2cxcywh(pred[:, :4])
            output_batch = torch.cat([pred[:, 6:7], bbox], dim=-1)
            if output_batch.shape[0] > 50:
                output_batch = output_batch[:50]

            output[i, : output_batch.shape[0]] = output_batch

    return output.cuda()


def box_regression_encoder(gt_bboxs, anchors):
    gt_ctr_x = gt_bboxs[:, 0]
    gt_ctr_y = gt_bboxs[:, 1]
    gt_widths = gt_bboxs[:, 2]
    gt_heights = gt_bboxs[:, 3]

    anchor_ctr_x = anchors[:, 0]
    anchor_ctr_y = anchors[:, 1]
    anchor_widths = anchors[:, 2]
    anchor_heights = anchors[:, 3]

    targets_dx = (gt_ctr_x - anchor_ctr_x) / anchor_widths
    targets_dy = (gt_ctr_y - anchor_ctr_y) / anchor_heights
    targets_dw = torch.log(gt_widths / anchor_widths)
    targets_dh = torch.log(gt_heights / anchor_heights)
    if len(targets_dx.shape) == 1:
        targets = torch.stack(
            (targets_dx, targets_dy, targets_dw, targets_dh), dim=1
        )
    else:
        targets = torch.cat(
            (targets_dx, targets_dy, targets_dw, targets_dh), dim=1
        )
    return targets


def box_regression_decoder(bboxs, anchors, xyxy=True):
    """
    bboxs (tensor): (N, num_cls * num_anchors)
    """
    bbox_xform_clip = math.log(1000.0 / 16)
    dx = bboxs[:, 0::4]
    dy = bboxs[:, 1::4]
    dw = bboxs[:, 2::4]
    dh = bboxs[:, 3::4]

    anchor_ctr_x = anchors[:, 0]
    anchor_ctr_y = anchors[:, 1]
    anchor_widths = anchors[:, 2]
    anchor_heights = anchors[:, 3]

    dw = torch.clamp(dw, max=bbox_xform_clip)
    dh = torch.clamp(dh, max=bbox_xform_clip)

    pred_ctr_x = dx * anchor_widths[:, None] + anchor_ctr_x[:, None]
    pred_ctr_y = dy * anchor_heights[:, None] + anchor_ctr_y[:, None]
    pred_w = torch.exp(dw) * anchor_widths[:, None]
    pred_h = torch.exp(dh) * anchor_heights[:, None]

    # Distance from center to box's corner.
    if xyxy:
        c_to_c_h = (
            torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device)
            * pred_h
        )
        c_to_c_w = (
            torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device)
            * pred_w
        )

        pred_boxes1 = pred_ctr_x - c_to_c_w
        pred_boxes2 = pred_ctr_y - c_to_c_h
        pred_boxes3 = pred_ctr_x + c_to_c_w
        pred_boxes4 = pred_ctr_y + c_to_c_h
        pred_boxes = torch.stack(
            (pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2
        ).flatten(1)
    else:
        pred_boxes = torch.stack(
            (pred_ctr_x, pred_ctr_y, pred_w, pred_h), dim=2
        ).flatten(1)
    return pred_boxes


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, img_shape[1])  # x1
        boxes[:, 1].clamp_(0, img_shape[0])  # y1
        boxes[:, 2].clamp_(0, img_shape[1])  # x2
        boxes[:, 3].clamp_(0, img_shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img_shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img_shape[0])  # y1, y2
