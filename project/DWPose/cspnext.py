# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Tuple

import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
from torch import Tensor
# from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.registry import MODELS
from mmdet.utils import ConfigType
# , OptConfigType
# , OptMultiConfig
from ..layers import CSPLayer
from .csp_darknet import SPPBottleneck
import todos

import pdb

@MODELS.register_module()
class CSPNeXt(BaseModule):
    """CSPNeXt backbone used in RTMDet.

    Args:
        arch (str): Architecture of CSPNeXt, from {P5, P6}.
            Defaults to P5.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Defaults to 0.5.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Defaults to (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Defaults to -1.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Defaults to False.
        arch_ovewrite (list): Overwrite default arch settings.
            Defaults to None.
        spp_kernel_sizes: (tuple[int]): Sequential of kernel sizes of SPP
            layers. Defaults to (5, 9, 13).
        channel_attention (bool): Whether to add channel attention in each
            stage. Defaults to True.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Dictionary to construct and
            config norm layer. Defaults to dict(type='BN', requires_grad=True).
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to dict(type='SiLU').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`]): Initialization config dict.
    """
    # From left to right:
    # in_channels, out_channels, num_blocks, add_identity, use_spp
    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, 1024, 3, False, True]],
        'P6': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, 768, 3, True, False],
               [768, 1024, 3, False, True]]
    }

    def __init__(
        self,
        arch: str = 'P5',
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        out_indices=(2, 3, 4),
        frozen_stages: int = -1,
        use_depthwise: bool = False,
        expand_ratio: float = 0.5,
        arch_ovewrite: dict = None,
        spp_kernel_sizes=(5, 9, 13),
        channel_attention: bool = True,
        conv_cfg = None,
        norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type='SiLU'),
        norm_eval: bool = False,
        init_cfg = dict(
            type='Kaiming',
            layer='Conv2d',
            a=math.sqrt(5),
            distribution='uniform',
            mode='fan_in',
            nonlinearity='leaky_relu')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        # init_cfg={'type': 'Pretrained', 'prefix': 'backbone.', 'checkpoint': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-l_udp-aic-coco_210e-256x192-273b7631_20230130.pth'}
        # arch = 'P5'
        # deepen_factor = 1.0
        # widen_factor = 1.0
        # out_indices = (4,)
        # frozen_stages = -1
        # use_depthwise = False
        # expand_ratio = 0.5
        # arch_ovewrite = None
        # spp_kernel_sizes = (5, 9, 13)
        # channel_attention = True
        # conv_cfg = None
        # norm_cfg = {'type': 'SyncBN'}
        # act_cfg = {'type': 'SiLU'}
        # norm_eval = False
        # init_cfg = {'type': 'Pretrained', 'prefix': 'backbone.', 'checkpoint': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-l_udp-aic-coco_210e-256x192-273b7631_20230130.pth'}


        arch_setting = self.arch_settings[arch]
        # arch_setting -- 
        #     [[64, 128, 3, True, False], 
        #      [128, 256, 6, True, False], 
        #      [256, 512, 6, True, False], 
        #      [512, 1024, 3, False, True]]        
        if arch_ovewrite: # False
            arch_setting = arch_ovewrite
        assert set(out_indices).issubset(
            i for i in range(len(arch_setting) + 1))
        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError('frozen_stages must be in range(-1, '
                             'len(arch_setting) + 1). But received '
                             f'{frozen_stages}')

        self.out_indices = out_indices # (4,)
        self.frozen_stages = frozen_stages # -1
        self.use_depthwise = use_depthwise # False
        self.norm_eval = norm_eval # False
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        # ==>  conv -- <class 'mmcv.cnn.bricks.conv_module.ConvModule'>

        self.stem = nn.Sequential(
            ConvModule(
                3,
                int(arch_setting[0][0] * widen_factor // 2),
                3,
                padding=1,
                stride=2,
                norm_cfg=norm_cfg, # {'type': 'SyncBN'}
                act_cfg=act_cfg), # {'type': 'SiLU'}
            ConvModule(
                int(arch_setting[0][0] * widen_factor // 2),
                int(arch_setting[0][0] * widen_factor // 2),
                3,
                padding=1,
                stride=1,
                norm_cfg=norm_cfg, # {'type': 'SyncBN'}
                act_cfg=act_cfg),
            ConvModule(
                int(arch_setting[0][0] * widen_factor // 2),
                int(arch_setting[0][0] * widen_factor),
                3,
                padding=1,
                stride=1,
                norm_cfg=norm_cfg, # {'type': 'SyncBN'}
                act_cfg=act_cfg)) # {'type': 'SiLU'}
        self.layers = ['stem']

        for i, (in_channels, out_channels, num_blocks, add_identity,
                use_spp) in enumerate(arch_setting):
            in_channels = int(in_channels * widen_factor)
            out_channels = int(out_channels * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)
            stage = []
            conv_layer = conv(
                in_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg, # None
                norm_cfg=norm_cfg, # {'type': 'SyncBN'}
                act_cfg=act_cfg) # {'type': 'SiLU'}
            # (Pdb) conv_layer
            # ConvModule(
            #   (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            #   (bn): SyncBatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            #   (activate): SiLU(inplace=True)
            # )
            stage.append(conv_layer)
            if use_spp: # True
                spp = SPPBottleneck(
                    out_channels,
                    out_channels,
                    kernel_sizes=spp_kernel_sizes,
                    conv_cfg=conv_cfg, # None
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                stage.append(spp)
                # (Pdb) spp
                # SPPBottleneck(
                #   (conv1): ConvModule(
                #     (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                #     (bn): SyncBatchNorm(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                #     (activate): SiLU(inplace=True)
                #   )
                #   (poolings): ModuleList(
                #     (0): MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
                #     (1): MaxPool2d(kernel_size=9, stride=1, padding=4, dilation=1, ceil_mode=False)
                #     (2): MaxPool2d(kernel_size=13, stride=1, padding=6, dilation=1, ceil_mode=False)
                #   )
                #   (conv2): ConvModule(
                #     (conv): Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                #     (bn): SyncBatchNorm(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                #     (activate): SiLU(inplace=True)
                #   )
                # )
            csp_layer = CSPLayer(
                out_channels,
                out_channels,
                num_blocks=num_blocks,
                add_identity=add_identity,
                use_depthwise=use_depthwise,
                use_cspnext_block=True,
                expand_ratio=expand_ratio,
                channel_attention=channel_attention,
                conv_cfg=conv_cfg, # None
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            stage.append(csp_layer)
            self.add_module(f'stage{i + 1}', nn.Sequential(*stage))
            self.layers.append(f'stage{i + 1}')
        # xxxx8888 pdb.set_trace()

    # def _freeze_stages(self) -> None:
    #     if self.frozen_stages >= 0:
    #         for i in range(self.frozen_stages + 1):
    #             m = getattr(self, self.layers[i])
    #             m.eval()
    #             for param in m.parameters():
    #                 param.requires_grad = False

    # def train(self, mode=True) -> None:
    #     super().train(mode)
    #     self._freeze_stages()
    #     if mode and self.norm_eval:
    #         for m in self.modules():
    #             if isinstance(m, _BatchNorm):
    #                 m.eval()

    def forward(self, x: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        # tensor [x] size: [1, 3, 384, 288] , min: -2.1179039478302 , max: 2.552854061126709
        outs = []

        # self.layers -- ['stem', 'stage1', 'stage2', 'stage3', 'stage4']
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        # self.out_indices -- (4,) ==> len(outs) == 1
        # tensor [outs[0]] size: [1, 1024, 12, 9] , min: -0.27846455574035645 , max: 20.5461483001709

        return tuple(outs)

if __name__ == '__main__':
    model = CSPNeXt()
    print(model)
    