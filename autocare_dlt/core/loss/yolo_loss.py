import torch
import torch.nn as nn

from autocare_dlt.core.loss import BCE_FocalLoss, IOUloss
from autocare_dlt.core.model.head.yolov5_head import check_anchor_order
from autocare_dlt.core.utils.boxes import xyxy2cxcywh


class YoloLoss(nn.Module):
    sort_obj_iou = False

    # Compute losses
    def __init__(self, anchors, num_classes, hyp, autobalance=False):
        super().__init__()

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        if anchors is None:
            anchors = [
                [10, 13, 16, 30, 33, 23],
                [30, 61, 62, 45, 59, 119],
                [116, 90, 156, 198, 373, 326],
            ]
        else:
            anchors = anchors
        self.stride = torch.tensor([256.0 / x for x in [32.0, 16.0, 8.0]])
        self.cp, self.cn = smooth_BCE(
            eps=hyp.get("label_smoothing", 0.0)
        )  # positive, negative BCE targets

        self.na = len(anchors[0]) // 2  # number of anchors
        self.nc = num_classes  # number of classes
        self.num_layers = len(anchors)  # number of layers
        self.device = device
        a = torch.tensor(anchors).float().view(self.num_layers, -1, 2)
        self.register_buffer("anchors", a)
        check_anchor_order(self)  # must be in pixel-space (not grid-space)
        self.anchors /= self.stride.view(-1, 1, 1)
        self.anchors = self.anchors.to(device)

        # Focal loss
        alpha = hyp["fl_alpha"]
        gamma = hyp["fl_gamma"]  # focal loss gamma
        if gamma > 0:
            BCEcls = BCE_FocalLoss(
                alpha=alpha,
                gamma=gamma,
                pos_weight=torch.tensor([hyp["cls_pw"]]),
            )
            BCEobj = BCE_FocalLoss(
                alpha=alpha,
                gamma=gamma,
                pos_weight=torch.tensor([hyp["obj_pw"]]),
            )
        else:
            BCEcls = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([hyp["cls_pw"]], device=device)
            )
            BCEobj = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([hyp["obj_pw"]], device=device)
            )
        self.iou_loss = IOUloss(reduction="mean", loss_type="ciou", xyxy=False)

        self.ssi = (
            list(self.stride).index(16) if autobalance else 0
        )  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = (
            BCEcls,
            BCEobj,
            1.0,
            hyp,
            autobalance,
        )

        self.balance = {3: [4.0, 1.0, 0.4]}.get(
            self.num_layers, [4.0, 1.0, 0.25, 0.06, 0.02]
        )  # P3-P7

    def forward(self, preds, targets, **kwargs):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(
            preds, targets
        )  # targets

        # Losses
        for i, pi in enumerate(preds):  # layer index, layer predictions
            pi = pi.to(self.device)
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(
                pi.shape[:4], dtype=pi.dtype, device=self.device
            )  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split(
                    (2, 2, 1, self.nc), 1
                )  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou_loss, iou = self.iou_loss(
                    pbox, tbox[i], return_iou=True
                )  # iou(prediction, target)
                lbox += iou_loss
                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou.squeeze(1)  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(
                        pcls, self.cn, device=self.device
                    )  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = (
                    self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
                )

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        bs = tobj.shape[0]  # batch size

        # return (lbox + lobj + lcls)[0] * bs / 4, (torch.cat((lbox, lobj, lcls))* bs / 4).detach()
        return {
            "reg_loss": lbox[0] * bs / 4,
            "obj_loss": lobj[0] * bs / 4,
            "cls_loss": lcls[0] * bs / 4,
        }

    def build_targets(self, preds, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, 0  # number of anchors and targets
        new_targets = []
        for i, t in enumerate(targets):
            if t["labels"] is None:
                label = torch.zeros(0, 6).to(self.device)
            else:
                label = torch.zeros(len(t["labels"]), 6).to(self.device)
                label[:, 1] = t["labels"].type(label.dtype)
                label[:, 2:] = xyxy2cxcywh(t["boxes"])
            nt += len(label)
            label[:, 0] = i
            new_targets.append(label)

        new_targets = torch.cat(new_targets, 0).to(self.device)
        targets = new_targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(
            7, device=self.device
        )  # normalized to gridspace gain
        ai = (
            torch.arange(na, device=self.device)
            .float()
            .view(na, 1)
            .repeat(1, nt)
        )  # same as .repeat_interleave(nt)
        targets = torch.cat(
            (targets.repeat(na, 1, 1), ai[..., None]), 2
        )  # append anchor indices

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=self.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.num_layers):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(preds[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = (
                    torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]
                )  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(
                4, 1
            )  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append(
                (
                    b,
                    a,
                    gj.clamp_(0, gain[3].long() - 1),
                    gi.clamp_(0, gain[2].long() - 1),
                )
            )  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch


def smooth_BCE(
    eps=0.1,
):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps
