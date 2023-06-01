import sys

from torch import nn

from .ctc_loss import STRCTCLoss
from .focal_loss import BCE_FocalLoss, CE_FocalLoss
from .iou_loss import IOUloss
from .lpr_loss import LPRLoss
from .pose_loss import JointsMSELoss
from .retinanet_loss import RetinaNetLoss
from .ssd_loss import SSDLoss
from .yolo_loss import YoloLoss
from .seg_loss import SegLoss

for module in filter(lambda x: "loss" in x.lower(), dir(nn)):
    setattr(sys.modules[__name__], module, getattr(nn, module))
