import os
import torch
import torch.nn as nn
from  torch.nn import  functional as F
from typing import Tuple, List
from .cspnext import CSPNeXt
from .rtmcc_head import RTMCCHead

import pdb


class DWPose(nn.Module):
    def __init__(self):
        super(DWPose, self).__init__()
        self.backbone = CSPNeXt()
        self.head = RTMCCHead()
        self.load_weights()


    def forward(self, x):
        B, C, H, W = x.size()
        # x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=True)

        return x


    def load_weights(self, model_path="models/DWPose.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path

        if os.path.exists(checkpoint):
            print(f"Loading weight from {checkpoint} ...")
            weight = torch.load(checkpoint) 
            self.load_state_dict(weight['state_dict'])
        else:
            print("-" * 32, "Warnning", "-" * 32)
            print(f"Weight file '{checkpoint}' not exist !!!")
