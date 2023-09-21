# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model import ImgDataPreprocessor

from mmpose.registry import MODELS


@MODELS.register_module()
class PoseDataPreprocessor(ImgDataPreprocessor):
    """Image pre-processor for pose estimation tasks."""


# class ImgDataPreprocessor(BaseDataPreprocessor):
#     """Image pre-processor for normalization and bgr to rgb conversion.
#     """
#     def __init__(self,
#                  mean: Optional[Sequence[Union[float, int]]] = None,
#                  std: Optional[Sequence[Union[float, int]]] = None,
#                  pad_size_divisor: int = 1,
#                  pad_value: Union[float, int] = 0,
#                  bgr_to_rgb: bool = False,
#                  rgb_to_bgr: bool = False,
#                  non_blocking: Optional[bool] = False):