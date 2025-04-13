# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from ...engine.config import Config
from ...engine.runner import load_checkpoint
from torch import Tensor

from mmdet.registry import MODELS
from ..structures import SampleList
from ..structures.bbox import cat_boxes
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, reduce_mean)
from ..utils import images_to_levels, multi_apply, unpack_gt_instances
from .single_stage_detector import SingleStageDetector

from ..backbones import ResNet
from ..necks.fpn import FPN
from ..dense_heads.retina_head import RetinaHead

from ..losses.crosskd.kd_quality_focal_loss import KDQualityFocalLoss
from ..losses.crosskd.giou_loss import GIoULoss


# @MODELS.register_module()
class CrossKDSingleStageDetector(SingleStageDetector):
    r"""Implementation of `Distilling the Knowledge in a Neural Network.
    <https://arxiv.org/abs/1503.02531>`_.

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone module.
        neck (:obj:`ConfigDict` or dict): The neck module.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head module.
        teacher_config (:obj:`ConfigDict` | dict | str | Path): Config file
            path or the config object of teacher model.
        teacher_ckpt (str, optional): Checkpoint path of teacher model.
            If left as None, the model will not load any weights.
            Defaults to True.
        eval_teacher (bool): Set the train mode for teacher.
            Defaults to True.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of ATSS. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of ATSS. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.


    Docs:
    teacher_config='configs/retinanet/retinanet_r50_fpn_1x_coco.py',
    teacher_ckpt=teacher_ckpt,
            
    
    kd_cfg=dict(
        loss_cls_kd=dict(type='KDQualityFocalLoss', beta=1, loss_weight=1.0),
        loss_reg_kd=dict(type='GIoULoss', loss_weight=1.0),
        reused_teacher_head_idx=3),

    
        will pass the teacher model (which is an object of Single stage detector).

        will hardcode kdconfig
    """

    def __init__(
            
        self,
        backbone: ResNet,
        neck: FPN,
        bbox_head: RetinaHead,
        teacher,
        teacher_ckpt,

        # teacher_config: Union[ConfigType, str, Path],
        # teacher_ckpt: Optional[str] = None,
        # kd_cfg: OptConfigType = None,
        # train_cfg: OptConfigType = None,
        # test_cfg: OptConfigType = None,
        # data_preprocessor: OptConfigType = None,
    ) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            # train_cfg=train_cfg,
            # test_cfg=test_cfg,
            # data_preprocessor=data_preprocessor
        )


        # Build teacher model (we will pass the builed teacher model)
        # if isinstance(teacher_config, (str, Path)):
        #     teacher_config = Config.fromfile(teacher_config)
        # self.teacher = MODELS.build(teacher_config['model'])

        self.teacher = teacher

        # TODO: if there is teacher checkpoint (has to load from it)
        # if teacher_ckpt is not None:
        #     load_checkpoint(self.teacher, teacher_ckpt, map_location='cpu')


        # In order to reforward teacher model,
        # set requires_grad of teacher model to False
        self.freeze(self.teacher)


        # self.loss_cls_kd = MODELS.build(kd_cfg['loss_cls_kd'])              
        # self.loss_reg_kd = MODELS.build(kd_cfg['loss_reg_kd'])             


        self.loss_cls_kd = KDQualityFocalLoss(
            beta=1,
            loss_weight=1.0
        )
        self.loss_reg_kd = GIoULoss(
            loss_weight=1.0
        )

        # self.with_feat_distill = False
        # if kd_cfg.get('loss_feat_kd', None):
        #     self.loss_feat_kd = MODELS.build(kd_cfg['loss_feat_kd'])
        #     self.with_feat_distill = True
        self.reused_teacher_head_idx = 3

    @staticmethod
    def freeze(model: nn.Module):
        """Freeze the model."""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def cuda(self, device: Optional[str] = None) -> nn.Module:
        """Since teacher is registered as a plain object, it is necessary to
        put the teacher model to cuda when calling ``cuda`` function."""
        self.teacher.cuda(device=device)
        return super().cuda(device=device)

    def to(self, device: Optional[str] = None) -> nn.Module:
        """Since teacher is registered as a plain object, it is necessary to
        put the teacher model to other device when calling ``to`` function."""
        self.teacher.to(device=device)
        return super().to(device=device)

    def train(self, mode: bool = True) -> None:
        """Set the same train mode for teacher and student model."""
        self.teacher.train(False)
        super().train(mode)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute, i.e. self.name = value

        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
        if name == 'teacher':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)
