# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
# from mmengine.dist import get_dist_info
# from mmengine.structures import PixelData
from torch import Tensor, nn

# from mmpose.codecs.utils import get_simcc_normalized
# from mmpose.evaluation.functional import simcc_pck_accuracy
from mmpose.models.utils.rtmcc_block import RTMCCBlock, ScaleNorm
from mmpose.models.utils.tta import flip_vectors
from mmpose.registry import KEYPOINT_CODECS, MODELS
# from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptSampleList)
from ..base_head import BaseHead

OptIntSeq = Optional[Sequence[int]]
import todos
import pdb

@MODELS.register_module()
class RTMCCHead(BaseHead):
    """Top-down head introduced in RTMPose (2023). The head is composed of a
    large-kernel convolutional layer, a fully-connected layer and a Gated
    Attention Unit to generate 1d representation from low-resolution feature
    maps.

    Args:
        in_channels (int | sequence[int]): Number of channels in the input
            feature map.
        out_channels (int): Number of channels in the output heatmap.
        input_size (tuple): Size of input image in shape [w, h].
        in_featuremap_size (int | sequence[int]): Size of input feature map.
        simcc_split_ratio (float): Split ratio of pixels.
            Default: 2.0.
        final_layer_kernel_size (int): Kernel size of the convolutional layer.
            Default: 1.
        gau_cfg (Config): Config dict for the Gated Attention Unit.
            Default: dict(
                hidden_dims=256,
                s=128,
                expansion_factor=2,
                dropout_rate=0.,
                drop_path=0.,
                act_fn='ReLU',
                use_rel_bias=False,
                pos_enc=False).
        loss (Config): Config of the keypoint loss. Defaults to use
            :class:`KLDiscretLoss`
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings
    """

    def __init__(
        self,
        in_channels: Union[int, Sequence[int]],
        out_channels: int,
        input_size: Tuple[int, int],
        in_featuremap_size: Tuple[int, int],
        simcc_split_ratio: float = 2.0,
        final_layer_kernel_size: int = 1,
        gau_cfg: ConfigType = dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='ReLU',
            use_rel_bias=False,
            pos_enc=False),
        loss: ConfigType = dict(type='KLDiscretLoss', use_target_weight=True),
        decoder: OptConfigType = None,
        init_cfg: OptConfigType = None,
    ):
        # init_cfg=[{'type': 'Normal', 'layer': ['Conv2d'], 'std': 0.001}, {'type': 'Constant', 'layer': 'BatchNorm2d', 'val': 1}, {'type': 'Normal', 'layer': ['Linear'], 'std': 0.01, 'bias': 0}]
        # in_channels = 1024
        # out_channels = 133
        # input_size = (288, 384)
        # in_featuremap_size = (9, 12)
        # simcc_split_ratio = 2.0
        # final_layer_kernel_size = 7
        # gau_cfg = {'hidden_dims': 256, 's': 128, 'expansion_factor': 2, 'dropout_rate': 0.0, 'drop_path': 0.0, 'act_fn': 'SiLU', 'use_rel_bias': False, 'pos_enc': False}
        # loss = {'type': 'KLDiscretLoss', 'use_target_weight': True, 'beta': 10.0, 'label_softmax': True}
        # decoder = {'type': 'SimCCLabel', 'input_size': (288, 384), 'sigma': (6.0, 6.93), 'simcc_split_ratio': 2.0, 'normalize': False, 'use_dark': False}
        # init_cfg = [{'type': 'Normal', 'layer': ['Conv2d'], 'std': 0.001}, {'type': 'Constant', 'layer': 'BatchNorm2d', 'val': 1}, {'type': 'Normal', 'layer': ['Linear'], 'std': 0.01, 'bias': 0}]

        # if init_cfg is None:
        #     init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.in_featuremap_size = in_featuremap_size
        self.simcc_split_ratio = simcc_split_ratio

        # self.loss_module = MODELS.build(loss)

        # xxxx8888
        if decoder is not None: # True
            self.decoder = KEYPOINT_CODECS.build(decoder) # <mmpose.codecs.simcc_label.SimCCLabel object>
        else:
            self.decoder = None

        if isinstance(in_channels, (tuple, list)):
            raise ValueError(
                f'{self.__class__.__name__} does not support selecting '
                'multiple input features.')

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
            dropout_rate=gau_cfg['dropout_rate'],
            drop_path=gau_cfg['drop_path'],
            attn_type='self-attn',
            act_fn=gau_cfg['act_fn'],
            use_rel_bias=gau_cfg['use_rel_bias'],
            pos_enc=gau_cfg['pos_enc'])

        self.cls_x = nn.Linear(gau_cfg['hidden_dims'], W, bias=False)
        self.cls_y = nn.Linear(gau_cfg['hidden_dims'], H, bias=False)
        # pdb.set_trace()

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor, Tensor]:
        """Forward the network.

        The input is the featuremap extracted by backbone and the
        output is the simcc representation.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            pred_x (Tensor): 1d representation of x.
            pred_y (Tensor): 1d representation of y.
        """
        feats = feats[-1]

        feats = self.final_layer(feats)  # -> B, K, H, W

        # flatten the output heatmap
        feats = torch.flatten(feats, 2)

        feats = self.mlp(feats)  # -> B, K, hidden

        feats = self.gau(feats)

        pred_x = self.cls_x(feats)
        pred_y = self.cls_y(feats)

        return pred_x, pred_y

    # xxxx8888
    def predict(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        test_cfg: OptConfigType = {},
    ) -> InstanceList:
        """Predict results from features.

        Args:
            feats (Tuple[Tensor] | List[Tuple[Tensor]]): The multi-stage
                features (or multiple multi-stage features in TTA)
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            test_cfg (dict): The runtime config for testing process. Defaults
                to {}

        Returns:
            List[InstanceData]: The pose predictions, each contains
            the following fields:
                - keypoints (np.ndarray): predicted keypoint coordinates in
                    shape (num_instances, K, D) where K is the keypoint number
                    and D is the keypoint dimension
                - keypoint_scores (np.ndarray): predicted keypoint scores in
                    shape (num_instances, K)
                - keypoint_x_labels (np.ndarray, optional): The predicted 1-D
                    intensity distribution in the x direction
                - keypoint_y_labels (np.ndarray, optional): The predicted 1-D
                    intensity distribution in the y direction
        """
        # test_cfg -- {'flip_test': True}

        if test_cfg.get('flip_test', False): # True
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            # flip_indices --
            # [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 20, 21, 22, 17, 18, 19, 39, 
            #  38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 49, 48, 47, 46, 45, 44, 
            #  43, 42, 41, 40, 50, 51, 52, 53, 58, 57, 56, 55, 54, 68, 67, 66, 65, 70, 69, 62, 61, 60, 
            #  59, 64, 63, 77, 76, 75, 74, 73, 72, 71, 82, 81, 80, 79, 78, 87, 86, 85, 84, 83, 90, 89, 
            #  88, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 
            #  129, 130, 131, 132, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 
            #  107, 108, 109, 110, 111]

            _feats, _feats_flip = feats

            _batch_pred_x, _batch_pred_y = self.forward(_feats)
            _batch_pred_x_flip, _batch_pred_y_flip = self.forward(_feats_flip)

            # xxxx3333
            todos.debug.output_var("head input1: ", _feats[0])
            todos.debug.output_var("head input2: ", _feats_flip[0])            

            _batch_pred_x_flip, _batch_pred_y_flip = flip_vectors(
                _batch_pred_x_flip,
                _batch_pred_y_flip,
                flip_indices=flip_indices)

            batch_pred_x = (_batch_pred_x + _batch_pred_x_flip) * 0.5
            batch_pred_y = (_batch_pred_y + _batch_pred_y_flip) * 0.5
        else:
            batch_pred_x, batch_pred_y = self.forward(feats)

        todos.debug.output_var("head midle output1: ", batch_pred_x)
        todos.debug.output_var("head midle output2: ", batch_pred_y)

        preds = self.decode((batch_pred_x, batch_pred_y)) # self.decoder.decode
        # type(preds[0]) -- <class 'mmengine.structures.instance_data.InstanceData'>

        todos.debug.output_var("head output1: ", preds[0].keypoints)
        todos.debug.output_var("head output2: ", preds[0].keypoint_scores)

        if test_cfg.get('output_heatmaps', False): # False
            pass
            # rank, _ = get_dist_info()
            # if rank == 0:
            #     warnings.warn('The predicted simcc values are normalized for '
            #                   'visualization. This may cause discrepancy '
            #                   'between the keypoint scores and the 1D heatmaps'
            #                   '.')

            # # normalize the predicted 1d distribution
            # batch_pred_x = get_simcc_normalized(batch_pred_x)
            # batch_pred_y = get_simcc_normalized(batch_pred_y)

            # B, K, _ = batch_pred_x.shape
            # # B, K, Wx -> B, K, Wx, 1
            # x = batch_pred_x.reshape(B, K, 1, -1)
            # # B, K, Wy -> B, K, 1, Wy
            # y = batch_pred_y.reshape(B, K, -1, 1)
            # # B, K, Wx, Wy
            # batch_heatmaps = torch.matmul(y, x)
            # pred_fields = [
            #     PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()
            # ]

            # for pred_instances, pred_x, pred_y in zip(preds,
            #                                           to_numpy(batch_pred_x),
            #                                           to_numpy(batch_pred_y)):

            #     pred_instances.keypoint_x_labels = pred_x[None]
            #     pred_instances.keypoint_y_labels = pred_y[None]

            # return preds, pred_fields
        else:
            return preds

    # xxxx8888
    # def loss(
    #     self,
    #     feats: Tuple[Tensor],
    #     batch_data_samples: OptSampleList,
    #     train_cfg: OptConfigType = {},
    # ) -> dict:
    #     """Calculate losses from a batch of inputs and data samples."""
    #     pass
    #     # pdb.set_trace()

    #     # pred_x, pred_y = self.forward(feats)

    #     # gt_x = torch.cat([
    #     #     d.gt_instance_labels.keypoint_x_labels for d in batch_data_samples
    #     # ],
    #     #                  dim=0)
    #     # gt_y = torch.cat([
    #     #     d.gt_instance_labels.keypoint_y_labels for d in batch_data_samples
    #     # ],
    #     #                  dim=0)
    #     # keypoint_weights = torch.cat(
    #     #     [
    #     #         d.gt_instance_labels.keypoint_weights
    #     #         for d in batch_data_samples
    #     #     ],
    #     #     dim=0,
    #     # )

    #     # pred_simcc = (pred_x, pred_y)
    #     # gt_simcc = (gt_x, gt_y)

    #     # # calculate losses
    #     # losses = dict()
    #     # loss = self.loss_module(pred_simcc, gt_simcc, keypoint_weights)

    #     # losses.update(loss_kpt=loss)

    #     # # calculate accuracy
    #     # _, avg_acc, _ = simcc_pck_accuracy(
    #     #     output=to_numpy(pred_simcc),
    #     #     target=to_numpy(gt_simcc),
    #     #     simcc_split_ratio=self.simcc_split_ratio,
    #     #     mask=to_numpy(keypoint_weights) > 0,
    #     # )

    #     # acc_pose = torch.tensor(avg_acc, device=gt_x.device)
    #     # losses.update(acc_pose=acc_pose)

    #     # return losses

    # xxxx8888
    # @property
    # def default_init_cfg(self):
    #     init_cfg = [
    #         dict(type='Normal', layer=['Conv2d'], std=0.001),
    #         dict(type='Constant', layer='BatchNorm2d', val=1),
    #         dict(type='Normal', layer=['Linear'], std=0.01, bias=0),
    #     ]
    #     return init_cfg
