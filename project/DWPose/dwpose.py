import os
import torch
import torch.nn as nn
from  torch.nn import functional as F
from torchvision import transforms as T

from typing import Tuple, List
from .cspnext import CSPNeXt
from .rtmcc_head import RTMCCHead
import todos

import pdb


class DWPose(nn.Module):
    def __init__(self):
        super(DWPose, self).__init__()
        self.backbone = CSPNeXt()
        self.head = RTMCCHead()
        self.bgr_normal = T.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        # self.normal = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.load_weights()

    def forward(self, x):
        B, C, H, W = x.size()
        # Convert RGB[0.0, 1.0] to BGR[0, 255] !!!
        x = torch.cat([x[:, 2:3, :, :], x[:, 1:2, :, :], x[:, 0:1, :, :]], dim = 1)
        x = self.bgr_normal(x)
        # x = self.normal(x)
        x = F.interpolate(x, size=(384, 288), mode="bilinear", align_corners=True)

        f1 = self.backbone(x)
        f2 = self.backbone(x.flip(-1)) # flip

        feats = [f1, f2]
        keypoints, scores = self.head(feats)

        keypoints[:, :, 0:1] = keypoints[:, :, 0:1]/288.0 * W
        keypoints[:, :, 1:2] = keypoints[:, :, 1:2]/384.0 * H
        # todos.debug.output_var("keypoints", keypoints)


        # todos.debug.output_var("scores", scores)


        # tensor [keypoints] size: [1, 133, 2] , min: 43.5 , max: 345.0
        # (Pdb) keypoints
        # tensor([[[163.0000,  72.0000],
        #          [167.5000,  67.5000],
        #          [159.5000,  68.0000],
        #          [174.5000,  72.5000],
        #          [158.5000,  76.0000],

        #          [177.5000,  98.5000],
        #          [165.0000, 113.0000],
        #          [168.5000,  69.0000],
        #          [157.0000, 145.5000],
        #          [166.5000,  51.5000],

        # OK ?????
        # array [results[0].pred_instances.keypoints] shape: (1, 133, 2) ,
        #       min: 72.22222757339478 , max: 497.7777777777777

        # keypoints: array([[[160. , 145.5],
        #             [163. , 142.5],
        #             [159. , 143. ],
        #             [168. , 145. ],
        #             [157.5, 146. ],

        #             [175. , 158. ],
        #             [159.5, 160.5],
        #             [185.5, 172.5],
        #             [155. , 172. ],
        #             [187.5, 175.5],
        #

        # OK ????
        # array [results[0].pred_instances.keypoint_scores] shape: (1, 133) , 
        #     min: 0.47906744 , max: 0.8998748    


        # results[0].pred_instances.keypoints ---
        # array([[[364.44444444,  83.33333826],
        #         [372.77777778,  75.00000525],
        #         [361.66666667,  76.38889408],
        #         [386.66666667,  81.94444942],
        #         [357.5       ,  84.7222271 ],
        #         [406.11111111, 118.05555916],
        #         [363.05555556, 125.00000334],
        #         [435.27777778, 158.3333354 ],



        # tensor [scores] size: [1, 133] , min: 0.2764616310596466 , max: 0.8369803428649902
        # (Pdb) scores
        # tensor([[0.5315, 0.5543, 0.5288, 0.5330, 0.4442, 0.5882, 0.5560, 0.2765, 0.6223,
        #          0.3811, 0.6604, 0.6192, 0.6640, 0.7669, 0.8077, 0.7530, 0.8370, 0.7147,


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
