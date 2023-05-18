import math

import numpy as np


def pck_accuracy(
    output: np.ndarray,
    target_dict: list,
):
    """
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    """
    idx = list(range(output.shape[1]))
    norm = 1.0
    # TODO: check get heatmap array correctly.
    target = np.zeros_like(output)
    for i, t in enumerate(target_dict):
        target[i] = t["heatmap"].detach().cpu().numpy()
    pred, _ = get_max_preds(output)
    target, _ = get_max_preds(target)
    h = output.shape[2]
    w = output.shape[3]
    norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm)

    acc = np.zeros(len(idx) + 1)
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    """Return percentage below threshold while ignoring values with a -1"""
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def convert_keypoints_to_coco(heatmaps, labels):
    preds, maxvals = get_final_preds(heatmaps, labels)
    preds = np.concatenate(preds, maxvals)
    # TODO: change preds shape (n, num_joints, 3) to (n, num_joints*3)

    results = []
    # TODO: write coco format using for statement
    #       result.append({
    #             "image_id": ,
    #             "category_id": ,
    #             "keypoints": ,
    #             "bbox": ,
    #         })

    return results


def get_final_preds(heatmaps, targets):

    coords, maxvals = get_max_preds(heatmaps)

    heatmap_height = heatmaps.shape[2]
    heatmap_width = heatmaps.shape[3]

    ##  post-processing
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = heatmaps[n][p]
            px = int(math.floor(coords[n][p][0] + 0.5))
            py = int(math.floor(coords[n][p][1] + 0.5))
            if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                diff = np.array(
                    [
                        hm[py][px + 1] - hm[py][px - 1],
                        hm[py + 1][px] - hm[py - 1][px],
                    ]
                )
                coords[n][p] += np.sign(diff) * 0.25

    preds = coords.copy()

    # TODO Transform back
    img_h, img_w = targets["img_size"]
    x, y, _, _ = targets["raw_box"]
    dx, dy = targets["pad"]
    rx, ry = targets["ratio"]
    preds[:, :, 0] = ((preds[:, :, 0] / heatmap_width * img_w) - dx) / rx + x
    preds[:, :, 1] = ((preds[:, :, 1] / heatmap_height * img_h) - dy) / ry + y

    return preds, maxvals


def get_max_preds(batch_heatmaps: np.ndarray):
    """
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    """

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals
