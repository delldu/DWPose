import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms as T

from .cspnext import CSPNeXt
from .rtmcc_head import RTMCCHead
from typing import Tuple, List

import todos

import pdb


class DWPose(nn.Module):
    def __init__(self, version="l384x288"):
        super(DWPose, self).__init__()
        self.version = version

        self.backbone = CSPNeXt(version=version)
        self.head = RTMCCHead(version=version)
        self.normal = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if version == "l384x288":
            self.std_h = 384
            self.std_w = 288
        else:
            self.std_h = 256
            self.std_w = 192

        self.load_weights()

    def forward(self, x):
        B, C, H, W = x.size()

        # x = F.interpolate(x, size=(384, 288), mode="bilinear", align_corners=True)
        x, s, tx, ty = self.image_transform(x)
        x = self.normal(x)

        # Convert RGB[0.0, 1.0] to BGR[0.0, 1.0] !!!
        x = torch.cat([x[:, 2:3, :, :], x[:, 1:2, :, :], x[:, 0:1, :, :]], dim = 1)

        f1 = self.backbone(x)
        f2 = self.backbone(x.flip(-1)) # flip

        feats = [f1, f2]
        keypoints, scores = self.head(feats)

        self.keypoint_transform(keypoints, s, tx, ty)
        output = torch.cat((keypoints, scores.unsqueeze(2)), dim=2)

        return output

    def image_transform(self, x) -> Tuple[torch.Tensor, float, int, int]:
        B, C, H, W = x.size()
        s = min(self.std_h/(1.25*H), self.std_w/(1.25 * W))
        # s = min(std_h*1.0/H, std_w*1.0/W)/1.25

        NH = int(s * H)
        NW = int(s * W)
        tx = (self.std_w - NW) // 2
        ty = (self.std_h - NH) // 2
        padding = (tx, ty, (self.std_w - NW) - tx, (self.std_h - NH) - ty) # left, top, right and bottom
        y = F.interpolate(x, size=(NH, NW), mode="bilinear", align_corners=True)
        y = T.functional.pad(y, padding, fill=0.0)

        # angle = 0.0
        # shear = 0.0
        # y = T.functional.affine(x, angle, [tx, ty], s, shear)
        return y, s, tx, ty

    def keypoint_transform(self, keypoints, s:float, tx:int, ty:int):
        keypoints[:, :, 0:1] = keypoints[:, :, 0:1] - tx
        keypoints[:, :, 0:1] = keypoints[:, :, 0:1] / s
        keypoints[:, :, 1:2] = keypoints[:, :, 1:2] - ty
        keypoints[:, :, 1:2] = keypoints[:, :, 1:2] / s


    def load_weights(self, model_path="models/DWPose-l.pth"):
        if self.version == "l384x288":
            model_path = "models/DWPose-l.pth"
        else:
            model_path = "models/DWPose-m.pth"
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path

        if os.path.exists(checkpoint):
            print(f"Loading weight from {checkpoint} ...")
            weight = torch.load(checkpoint) 
            self.load_state_dict(weight['state_dict'])
        else:
            print("-" * 32, "Warnning", "-" * 32)
            print(f"Weight file '{checkpoint}' not exist !!!")
