# Copyright (c) OpenMMLab. All rights reserved.

import warnings
import math
import torch
import torch.nn as nn
from  torch.nn import functional as F
# import numpy as np

from typing import Tuple
import todos
import pdb

def flip_vectors(x_labels, y_labels):
    """Flip instance-level labels in specific axis for test-time augmentation.
    Args:
        x_labels: The vector labels in x-axis to flip. Should be in shape [B, C, Wx]
        y_labels: The vector labels in y-axis to flip. Should be in shape [B, C, Wy]
    """

    flip_indices = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 20, 21, 
                    22, 17, 18, 19, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 
                    26, 25, 24, 23, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 50, 51, 52,
                    53, 58, 57, 56, 55, 54, 68, 67, 66, 65, 70, 69, 62, 61, 60, 59, 64,
                    63, 77, 76, 75, 74, 73, 72, 71, 82, 81, 80, 79, 78, 87, 86, 85, 84,
                    83, 90, 89, 88, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
                    123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 91, 92, 93, 94, 95,
                    96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]

    assert x_labels.ndim == 3 and y_labels.ndim == 3
    assert len(flip_indices) == x_labels.shape[1] and len(flip_indices) == y_labels.shape[1]

    x_labels = x_labels[:, flip_indices].flip(-1)
    y_labels = y_labels[:, flip_indices]

    return x_labels, y_labels

def get_simcc_maximum(simcc_x, simcc_y) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get maximum response location and value from simcc representations.
    Args:
        simcc_x: x-axis SimCC in shape (N, K, Wx)
        simcc_y: y-axis SimCC in shape (N, K, Wy)
    Returns:
        - locs: locations of maximum heatmap responses in shape (N, K, 2)
        - vals: values of maximum heatmap responses in shape (N, K)
    """
    # xxxx5555

    # todos.debug.output_var("simcc_x", simcc_x)
    # todos.debug.output_var("simcc_y", simcc_y)

    N, K, Wx = simcc_x.size() # (1, 133, 576)
    simcc_x = simcc_x.reshape(N * K, -1)
    simcc_y = simcc_y.reshape(N * K, -1)

    x_vals, x_locs = torch.max(simcc_x, dim=1) # size() -- [133]
    y_vals, y_locs = torch.max(simcc_y, dim=1) # size() -- [133]
    locs = torch.stack((x_locs, y_locs), dim=1) # [133, 2]
    # --------------------------------------------------------------

    # max_val_x = torch.max(simcc_x, dim=1)
    # max_val_y = torch.max(simcc_y, dim=1)

    mask = x_vals > y_vals
    x_vals[mask] = y_vals[mask]
    vals = x_vals
    locs[vals <= 0.] = -1

    locs = locs.reshape(N, K, 2)
    vals = vals.reshape(N, K)

    # todos.debug.output_var("locs", locs)
    # todos.debug.output_var("vals", vals)

    # Bad ---
    # tensor [simcc_x] size: [1, 133, 576] , min: -0.5505505204200745 , max: 0.8369803428649902
    # tensor [simcc_y] size: [1, 133, 768] , min: -0.39678260684013367 , max: 0.9399464130401611
    # tensor [locs] size: [1, 133, 2] , min: 87 , max: 690
    # tensor [vals] size: [1, 133] , min: 0.2764616310596466 , max: 0.8369803428649902


    # OK ----------
    # array [simcc_x] shape: (1, 133, 576) , min: -0.41311845 , max: 0.9175705
    # array [simcc_y] shape: (1, 133, 768) , min: -0.3252464 , max: 0.90258455
    # array [locs] shape: (1, 133, 2) , min: 265.0 , max: 509.0
    # array [vals] shape: (1, 133) , min: 0.47906744 , max: 0.8998748


    return locs, vals


class SimCCLabel(nn.Module):
    r"""Generate keypoint representation via "SimCC" approach.
    
    See the paper: `SimCC: a Simple Coordinate Classification Perspective for
    Human Pose Estimation`_ by Li et al (2022) for more details.
    Old name: SimDR

    https://arxiv.org/abs/2107.03332

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - image size: [w, h]

    """
    def __init__(self, simcc_split_ratio=2.0):
        super().__init__()
        self.simcc_split_ratio = simcc_split_ratio

    def forward(self, simcc_x, simcc_y) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode keypoint coordinates from SimCC representations.
        """
        keypoints, scores = get_simcc_maximum(simcc_x, simcc_y)
        # tensor [keypoints] size: [1, 133, 2] , min: 87 , max: 690
        # tensor [scores] size: [1, 133] , min: 0.2764616310596466 , max: 0.8369803428649902

        keypoints = keypoints/self.simcc_split_ratio # self.simcc_split_ratio -- 2.0
        # tensor [keypoints] size: [1, 133, 2] , min: 43.5 , max: 345.0

        return keypoints, scores


class Scale(nn.Module):
    """Scale vector by element multiplications. """

    def __init__(self, dim, init_value=1., trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


class ScaleNorm(nn.Module):
    """Scale Norm.
    Reference:
        `Transformers without Tears: Improving the Normalization
        of Self-Attention <https://arxiv.org/abs/1910.05895>`_
    """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim=2, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class RTMCCBlock(nn.Module):
    """Gated Attention Unit (GAU) in RTMBlock.
    Reference:
        `Transformer Quality in Linear Time
        <https://arxiv.org/abs/2202.10447>`_
    """

    def __init__(self,
                 num_token=133,
                 in_token_dims=256,
                 out_token_dims=256,
                 expansion_factor=2,
                 s=128,
                ):
        super(RTMCCBlock, self).__init__()
        self.s = s
        self.num_token = num_token
        self.e = int(in_token_dims * expansion_factor)
        self.o = nn.Linear(self.e, out_token_dims, bias=False)
        self.uv = nn.Linear(in_token_dims, 2 * self.e + self.s, bias=False)
        self.gamma = nn.Parameter(torch.rand((2, self.s)))
        self.beta = nn.Parameter(torch.rand((2, self.s)))
        self.ln = ScaleNorm(in_token_dims, eps=1e-05)
        self.act_fn = nn.SiLU(True)
        self.res_scale = Scale(in_token_dims)
        self.sqrt_s = math.sqrt(s)

    def _forward(self, inputs):
        """GAU Forward function."""
        x = inputs
        x = self.ln(x)

        # [B, K, in_token_dims] -> [B, K, e + e + s]
        uv = self.uv(x)
        uv = self.act_fn(uv)

        # [B, K, e + e + s] -> [B, K, e], [B, K, e], [B, K, s]
        u, v, base = torch.split(uv, [self.e, self.e, self.s], dim=2)
        # [B, K, 1, s] * [1, 1, 2, s] + [2, s] -> [B, K, 2, s]
        base = base.unsqueeze(2) * self.gamma[None, None, :] + self.beta

        # [B, K, 2, s] -> [B, K, s], [B, K, s]
        q, k = torch.unbind(base, dim=2)

        # [B, K, s].permute() -> [B, s, K]
        # [B, K, s] x [B, s, K] -> [B, K, K]
        qk = torch.bmm(q, k.permute(0, 2, 1))

        # [B, K, K]
        kernel = torch.square(F.relu(qk / self.sqrt_s))

        # [B, K, K] x [B, K, e] -> [B, K, e]
        x = u * torch.bmm(kernel, v)
        # [B, K, e] -> [B, K, out_token_dims]
        x = self.o(x)

        return x

    def forward(self, x):
        res_shortcut = x
        main_branch = self._forward(x)
        return self.res_scale(res_shortcut) + main_branch



class RTMCCHead(nn.Module):
    """Top-down head introduced in RTMPose (2023). The head is composed of a
    large-kernel convolutional layer, a fully-connected layer and a Gated
    Attention Unit to generate 1d representation from low-resolution feature
    maps.
    """

    def __init__(self,
        in_channels=1024,
        out_channels=133,
        input_size=(288, 384),
        in_featuremap_size=(9, 12),
        simcc_split_ratio=2.0,
        final_layer_kernel_size=7,
    ):
        super().__init__()

        gau_cfg = {'hidden_dims': 256, 's': 128, 'expansion_factor': 2, }

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.in_featuremap_size = in_featuremap_size
        self.simcc_split_ratio = simcc_split_ratio

        self.decoder = SimCCLabel(simcc_split_ratio=self.simcc_split_ratio)

        # Define SimCC layers
        flatten_dims = self.in_featuremap_size[0] * self.in_featuremap_size[1]

        self.final_layer = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=final_layer_kernel_size,
            stride=1,
            padding=final_layer_kernel_size // 2)
        self.mlp = nn.Sequential(
            ScaleNorm(flatten_dims),
            nn.Linear(flatten_dims, gau_cfg['hidden_dims'], bias=False))

        W = int(self.input_size[0] * self.simcc_split_ratio)
        H = int(self.input_size[1] * self.simcc_split_ratio)

        self.gau = RTMCCBlock(
            self.out_channels,
            gau_cfg['hidden_dims'],
            gau_cfg['hidden_dims'],
            s=gau_cfg['s'],
            expansion_factor=gau_cfg['expansion_factor'],
        )

        self.cls_x = nn.Linear(gau_cfg['hidden_dims'], W, bias=False)
        self.cls_y = nn.Linear(gau_cfg['hidden_dims'], H, bias=False)

    def forward(self, feats: Tuple[torch.Tensor]):
        # TTA: flip test -> feats = [orig, flipped]
        assert isinstance(feats, list) and len(feats) == 2
        # xxxx3333
        f1, f2 = feats
        todos.debug.output_var("head input1: ", f1)
        todos.debug.output_var("head input2: ", f1)


        pred_x, pred_y = self.forward_x(f1)

        pred_x_flip, pred_y_flip = self.forward_x(f2)
        pred_x_flip, pred_y_flip = flip_vectors(pred_x_flip, pred_y_flip)

        batch_pred_x = (pred_x + pred_x_flip) * 0.5
        batch_pred_y = (pred_y + pred_y_flip) * 0.5

        # todos.debug.output_var("batch_pred_x", batch_pred_x)
        # todos.debug.output_var("batch_pred_y", batch_pred_y)
        # tensor [batch_pred_x] size: [1, 133, 576] , min: -0.5505505204200745 , max: 0.8369803428649902
        # tensor [batch_pred_y] size: [1, 133, 768] , min: -0.39678260684013367 , max: 0.9399464130401611

        todos.debug.output_var("head midle output1: ", batch_pred_x)
        todos.debug.output_var("head midle output2: ", batch_pred_y)


        keypoints, scores = self.decoder(batch_pred_x, batch_pred_y)

        todos.debug.output_var("head output1: ", keypoints)
        todos.debug.output_var("head output2: ", scores)

        return keypoints, scores

    def forward_x(self, feats) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.final_layer(feats)  # -> B, K, H, W

        # flatten the output heatmap
        feats = torch.flatten(feats, 2)
        feats = self.mlp(feats)  # -> B, K, hidden
        feats = self.gau(feats)

        pred_x = self.cls_x(feats)
        pred_y = self.cls_y(feats)

        return pred_x, pred_y



if __name__ == '__main__':
	model = RTMCCHead()
	print(model)