# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Tuple, Union

import torch
from mmengine.model import BaseModel
from torch import Tensor

from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.models.utils import check_and_update_config
from mmpose.registry import MODELS
from mmpose.utils.typing import (ConfigType, ForwardResults, OptConfigType,
                                 Optional, OptMultiConfig, OptSampleList,
                                 SampleList)
import todos
import pdb

class BasePoseEstimator(BaseModel, metaclass=ABCMeta):
    """Base class for pose estimators.

    Args:
        data_preprocessor (dict | ConfigDict, optional): The pre-processing
            config of :class:`BaseDataPreprocessor`. Defaults to ``None``
        init_cfg (dict | ConfigDict): The model initialization config.
            Defaults to ``None``
        metainfo (dict): Meta information for dataset, such as keypoints
            definition and properties. If set, the metainfo of the input data
            batch will be overridden. For more details, please refer to
            https://mmpose.readthedocs.io/en/latest/user_guides/
            prepare_datasets.html#create-a-custom-dataset-info-
            config-file-for-the-dataset. Defaults to ``None``
    """
    _version = 2

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 metainfo: Optional[dict] = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        # backbone = {'_scope_': 'mmdet', 'type': 'CSPNeXt', 'arch': 'P5', 'expand_ratio': 0.5, 'deepen_factor': 1.0, 'widen_factor': 1.0, 'out_indices': (4,), 'channel_attention': True, 'norm_cfg': {'type': 'SyncBN'}, 'act_cfg': {'type': 'SiLU'}, 'init_cfg': {'type': 'Pretrained', 'prefix': 'backbone.', 'checkpoint': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-l_udp-aic-coco_210e-256x192-273b7631_20230130.pth'}}
        # neck = None
        # head = {'type': 'RTMCCHead', 'in_channels': 1024, 'out_channels': 133, 'input_size': (288, 384), 'in_featuremap_size': (9, 12), 'simcc_split_ratio': 2.0, 'final_layer_kernel_size': 7, 'gau_cfg': {'hidden_dims': 256, 's': 128, 'expansion_factor': 2, 'dropout_rate': 0.0, 'drop_path': 0.0, 'act_fn': 'SiLU', 'use_rel_bias': False, 'pos_enc': False}, 'loss': {'type': 'KLDiscretLoss', 'use_target_weight': True, 'beta': 10.0, 'label_softmax': True}, 'decoder': {'type': 'SimCCLabel', 'input_size': (288, 384), 'sigma': (6.0, 6.93), 'simcc_split_ratio': 2.0, 'normalize': False, 'use_dark': False}}
        # train_cfg = None
        # test_cfg = {'flip_test': True, 'output_heatmaps': True}
        # data_preprocessor = {'type': 'PoseDataPreprocessor', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'bgr_to_rgb': True}
        # init_cfg = None
        # metainfo = None

        
        self.metainfo = self._load_metainfo(metainfo) # None

        self.backbone = MODELS.build(backbone) # CSPNeXt(...)

        # the PR #2108 and #2126 modified the interface of neck and head.
        # The following function automatically detects outdated
        # configurations and updates them accordingly, while also providing
        # clear and concise information on the changes made.
        neck, head = check_and_update_config(neck, head)

        if neck is not None: # False
            self.neck = MODELS.build(neck)

        if head is not None: # True
            self.head = MODELS.build(head) # RTMCCHead

        self.train_cfg = train_cfg if train_cfg else {}
        self.test_cfg = test_cfg if test_cfg else {}

        # Register the hook to automatically convert old version state dicts
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)
        # pdb.set_trace()

    @property
    def with_neck(self) -> bool:
        """bool: whether the pose estimator has a neck."""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_head(self) -> bool:
        """bool: whether the pose estimator has a head."""
        return hasattr(self, 'head') and self.head is not None

    @staticmethod
    def _load_metainfo(metainfo: dict = None) -> dict:
        """Collect meta information from the dictionary of meta.

        Args:
            metainfo (dict): Raw data of pose meta information.

        Returns:
            dict: Parsed meta information.
        """

        if metainfo is None:
            return None

        if not isinstance(metainfo, dict):
            raise TypeError(
                f'metainfo should be a dict, but got {type(metainfo)}')

        metainfo = parse_pose_metainfo(metainfo)
        return metainfo

    def forward(self,
                inputs: torch.Tensor,
                data_samples: OptSampleList,
                mode: str = 'tensor') -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: 'tensor', 'predict' and 'loss':

        - 'tensor': Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - 'predict': Forward and return the predictions, which are fully
        processed to a list of :obj:`PoseDataSample`.
        - 'loss': Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general
            data_samples (list[:obj:`PoseDataSample`], optional): The
                annotation of every sample. Defaults to ``None``
            mode (str): Set the forward mode and return value type. Defaults
                to ``'tensor'``

        Returns:
            The return type depends on ``mode``.

            - If ``mode='tensor'``, return a tensor or a tuple of tensors
            - If ``mode='predict'``, return a list of :obj:``PoseDataSample``
                that contains the pose predictions
            - If ``mode='loss'``, return a dict of tensor(s) which is the loss
                function value
        """
        # mode === 'predict'
        # tensor [inputs] size: [1, 3, 384, 288] , min: -2.1179039478302 , max: 2.552854061126709

        if isinstance(inputs, list):
            inputs = torch.stack(inputs)
        if mode == 'loss':
            return None # self.loss(inputs, data_samples)
        elif mode == 'predict': # True
            # use customed metainfo to override the default metainfo
            # if self.metainfo is not None: # False
            #     for data_sample in data_samples:
            #         data_sample.set_metainfo(self.metainfo)
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode.')

    # @abstractmethod
    # def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
    #     """Calculate losses from a batch of inputs and data samples."""

    @abstractmethod
    def predict(self, inputs: Tensor, data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing."""

    # def _forward(self,
    #              inputs: Tensor,
    #              data_samples: OptSampleList = None
    #              ) -> Union[Tensor, Tuple[Tensor]]:
    #     """Network forward process. Usually includes backbone, neck and head
    #     forward without any post-processing.

    #     Args:
    #         inputs (Tensor): Inputs with shape (N, C, H, W).

    #     Returns:
    #         Union[Tensor | Tuple[Tensor]]: forward output of the network.
    #     """
    #     pdb.set_trace()
        
    #     x = self.extract_feat(inputs)
    #     if self.with_head:
    #         x = self.head.forward(x)

    #     return x

    def extract_feat(self, inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have various
            resolutions.
        """
        x = self.backbone(inputs)
        # if self.with_neck: # False
        #     x = self.neck(x)

        return x

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args,
                                  **kwargs):
        """A hook function to convert old-version state dict of
        :class:`TopdownHeatmapSimpleHead` (before MMPose v1.0.0) to a
        compatible format of :class:`HeatmapHead`.

        The hook will be automatically registered during initialization.
        """
        version = local_meta.get('version', None)
        if version and version >= self._version:
            return

        # convert old-version state dict
        keys = list(state_dict.keys())
        for k in keys:
            if 'keypoint_head' in k:
                v = state_dict.pop(k)
                k = k.replace('keypoint_head', 'head')
                state_dict[k] = v
