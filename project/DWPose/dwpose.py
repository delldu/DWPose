import os
import torch
import torch.nn as nn
from  torch.nn import  functional as F
from typing import Tuple, List

import pdb


class DWPose(nn.Module):
    def __init__(self):
        super(DWPose, self).__init__()

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
            self.load_state_dict(torch.load(checkpoint))
        else:
            print("-" * 32, "Warnning", "-" * 32)
            print(f"Weight file '{checkpoint}' not exist !!!")
