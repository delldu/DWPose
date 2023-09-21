# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
import todos
# from typing import Tuple

import pdb

class _BatchNormXd(nn.modules.batchnorm._BatchNorm):
    """A general BatchNorm layer without input dimension check.

    Reproduced from @kapily's work:
    (https://github.com/pytorch/pytorch/issues/41081#issuecomment-783961547)
    The only difference between BatchNorm1d, BatchNorm2d, BatchNorm3d, etc
    is `_check_input_dim` that is designed for tensor sanity checks.
    The check has been bypassed in this class for the convenience of converting
    SyncBatchNorm.
    """

    def _check_input_dim(self, input):
        return

class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 
            kernel_size=(kernel_size, kernel_size), 
            stride=(stride, stride), 
            padding=(padding, padding),
            groups=groups, 
            bias=False)
        # layer = nn.SyncBatchNorm(out_channels, eps=1e-05, momentum=0.1, 
        #     affine=True, track_running_stats=True)
        layer = _BatchNormXd(out_channels)

        self.add_module('bn', layer)
        self.activate = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activate(x)
        return x


class DepthwiseSeparableConvModule(nn.Module):
    """Depthwise separable convolution module.
    See https://arxiv.org/pdf/1704.04861.pdf for details.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                ):
        super().__init__()
        self.depthwise_conv = ConvModule(in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels)
        self.pointwise_conv = ConvModule(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes=(5, 9, 13),
                ):
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = ConvModule(in_channels, mid_channels, 1, stride=1)
        self.poolings = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes
        ])
        conv2_channels = mid_channels * (len(kernel_sizes) + 1)
        self.conv2 = ConvModule(conv2_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        with torch.cuda.amp.autocast(enabled=False):
            x = torch.cat(
                [x] + [pooling(x) for pooling in self.poolings], dim=1)
        x = self.conv2(x)
        return x

class ChannelAttention(nn.Module):
    """Channel attention Module.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            out = self.global_avgpool(x)
        out = self.fc(out)
        out = self.act(out)
        return x * out


class CSPNeXtBlock(nn.Module):
    """The basic bottleneck block used in CSPNeXt.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion: float = 0.5,
                 add_identity: bool = True,
                 kernel_size: int = 5,
                ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvModule(in_channels, hidden_channels, 3,
            stride=1, padding=1)
        self.conv2 = DepthwiseSeparableConvModule(
            hidden_channels, out_channels, kernel_size,
            stride=1, padding=kernel_size // 2)
        self.add_identity = add_identity and in_channels == out_channels # True or False

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out

class CSPLayer(nn.Module):
    """Cross Stage Partial Layer.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: float = 0.5,
                 num_blocks: int = 1,
                 add_identity: bool = True,
                 use_cspnext_block: bool = True,
                 channel_attention: bool = False,
                ):
        super().__init__()
        block = CSPNeXtBlock
        mid_channels = int(out_channels * expand_ratio)
        self.channel_attention = channel_attention
        self.main_conv = ConvModule(in_channels, mid_channels, 1)
        self.short_conv = ConvModule(in_channels, mid_channels, 1)
        self.final_conv = ConvModule(2 * mid_channels, out_channels, 1)

        self.blocks = nn.Sequential(*[
            block(mid_channels, mid_channels, 1.0, add_identity) for _ in range(num_blocks)
        ])
        if channel_attention:
            self.attention = ChannelAttention(2 * mid_channels)

    def forward(self, x):
        x_short = self.short_conv(x)

        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)

        x_final = torch.cat((x_main, x_short), dim=1)

        if self.channel_attention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final)


class CSPNeXt(nn.Module):
    """CSPNeXt backbone used in RTMDet.
    """
    # in_channels, out_channels, num_blocks, add_identity, use_spp
    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, 1024, 3, False, True]],
        'P6': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, 768, 3, True, False],
               [768, 1024, 3, False, True]]
    }

    def __init__(self,
        arch='P5',
        deepen_factor=1.0,
        widen_factor=1.0,
        out_indices=(4,),
        expand_ratio=0.5,
        spp_kernel_sizes=(5, 9, 13),
        channel_attention=True,
    ):
        super().__init__()

        # for "m256x192":
        # deepen_factor = 0.67
        # widen_factor = 0.75


        arch_setting = self.arch_settings[arch]
        # arch_setting -- 
        #     [[64, 128, 3, True, False], 
        #      [128, 256, 6, True, False], 
        #      [256, 512, 6, True, False], 
        #      [512, 1024, 3, False, True]]        
        assert set(out_indices).issubset(i for i in range(len(arch_setting) + 1))

        self.out_indices = out_indices # (4,)
        self.stem = nn.Sequential(
            ConvModule(3, int(arch_setting[0][0] * widen_factor // 2), 3,
                padding=1, stride=2),
            ConvModule(
                int(arch_setting[0][0] * widen_factor // 2),
                int(arch_setting[0][0] * widen_factor // 2),
                3,
                padding=1, stride=1),
            ConvModule(
                int(arch_setting[0][0] * widen_factor // 2),
                int(arch_setting[0][0] * widen_factor),
                3,
                padding=1, stride=1),
            )
        self.layers = ['stem']

        for i, (in_channels, out_channels, num_blocks, add_identity,
                use_spp) in enumerate(arch_setting):
            in_channels = int(in_channels * widen_factor)
            out_channels = int(out_channels * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)
            stage = []
            conv_layer = ConvModule(in_channels, out_channels, 3, stride=2, padding=1)
            stage.append(conv_layer)
            if use_spp: # True
                spp = SPPBottleneck(
                    out_channels,
                    out_channels,
                    kernel_sizes=spp_kernel_sizes,
                    )
                stage.append(spp)
            csp_layer = CSPLayer(
                out_channels,
                out_channels,
                num_blocks=num_blocks,
                add_identity=add_identity,
                use_cspnext_block=True,
                expand_ratio=expand_ratio,
                channel_attention=channel_attention,
                )
            stage.append(csp_layer)
            self.add_module(f'stage{i + 1}', nn.Sequential(*stage))
            self.layers.append(f'stage{i + 1}')

    def forward(self, x):
        # tensor [x] size: [1, 3, 384, 288] , min: -2.1179039478302 , max: 2.552854061126709

        # outs = []
        # for i, layer_name in enumerate(self.layers):
        #     layer = getattr(self, layer_name)
        #     x = layer(x)
        #     if i in self.out_indices:
        #         outs.append(x)
        # # self.out_indices -- (4,) ==> len(outs) == 1
        # # tensor [outs[0]] size: [1, 1024, 12, 9] , min: -0.27846455574035645 , max: 20.5461483001709
        # return outs[0]

        # self.layers -- ['stem', 'stage1', 'stage2', 'stage3', 'stage4']
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        return x

if __name__ == '__main__':
    model = CSPNeXt()
    print(model)
