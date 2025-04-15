# Copyright (c) OpenMMLab. All rights reserved.
from kd.models.backbones.resnet import ResNet
from kd.models.data_preprocessors.data_preprocessor import DetDataPreprocessor
from kd.models.dense_heads.retina_head import RetinaHead
from kd.models.necks.fpn import FPN
from kd.models.task_modules.assigners.max_iou_assigner import MaxIoUAssigner
from kd.models.task_modules.samplers.pseudo_sampler import PseudoSampler
# from ..registry import MODELS
from ..detutils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage_detector import SingleStageDetector


# @MODELS.register_module()
class RetinaNet(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone: ResNet,
                 neck: FPN,
                 bbox_head: RetinaHead,
                 data_preprocessor: DetDataPreprocessor,
                 train_cfg: dict,
                 test_cfg: dict,
                 checkpoint: str = "",
                #  init_cfg: OptMultiConfig = None
                 ) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            # init_cfg=init_cfg,
            checkpoint=checkpoint)