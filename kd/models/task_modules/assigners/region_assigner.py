# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
from ....engine.structures import InstanceData
from torch import Tensor

from ...registry import TASK_UTILS
from ..prior_generators import anchor_inside_flags
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


def calc_region(
        bbox: Tensor,
        ratio: float,
        stride: int,
        featmap_size: Optional[Tuple[int, int]] = None) -> Tuple[Tensor]:
    """Calculate region of the box defined by the ratio, the ratio is from the
    center of the box to every edge."""
    # project bbox on the feature
    f_bbox = bbox / stride
    x1 = torch.round((1 - ratio) * f_bbox[0] + ratio * f_bbox[2])
    y1 = torch.round((1 - ratio) * f_bbox[1] + ratio * f_bbox[3])
    x2 = torch.round(ratio * f_bbox[0] + (1 - ratio) * f_bbox[2])
    y2 = torch.round(ratio * f_bbox[1] + (1 - ratio) * f_bbox[3])
    if featmap_size is not None:
        x1 = x1.clamp(min=0, max=featmap_size[1])
        y1 = y1.clamp(min=0, max=featmap_size[0])
        x2 = x2.clamp(min=0, max=featmap_size[1])
        y2 = y2.clamp(min=0, max=featmap_size[0])
    return (x1, y1, x2, y2)


def anchor_ctr_inside_region_flags(anchors: Tensor, stride: int,
                                   region: Tuple[Tensor]) -> Tensor:
    """Get the flag indicate whether anchor centers are inside regions."""
    x1, y1, x2, y2 = region
    f_anchors = anchors / stride
    x = (f_anchors[:, 0] + f_anchors[:, 2]) * 0.5
    y = (f_anchors[:, 1] + f_anchors[:, 3]) * 0.5
    flags = (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)
    return flags


@TASK_UTILS.register_module()
class RegionAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, `0`, or a positive integer
    indicating the ground truth index.

    - -1: don't care
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        center_ratio (float): ratio of the region in the center of the bbox to
            define positive sample.
        ignore_ratio (float): ratio of the region to define ignore samples.
    """

    def __init__(self,
                 center_ratio: float = 0.2,
                 ignore_ratio: float = 0.5) -> None:
        self.center_ratio = center_ratio
        self.ignore_ratio = ignore_ratio

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               img_meta: dict,
               featmap_sizes: List[Tuple[int, int]],
               num_level_anchors: List[int],
               anchor_scale: int,
               anchor_strides: List[int],
               gt_instances_ignore: Optional[InstanceData] = None,
               allowed_border: int = 0) -> AssignResult:
        """Assign gt to anchors.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, 0, or a positive number. -1 means don't care,
        0 means negative sample, positive number is the index (1-based) of
        assigned gt.

        The assignment is done in following steps, and the order matters.

        1. Assign every anchor to 0 (negative)
        2. (For each gt_bboxes) Compute ignore flags based on ignore_region
           then assign -1 to anchors w.r.t. ignore flags
        3. (For each gt_bboxes) Compute pos flags based on center_region then
           assign gt_bboxes to anchors w.r.t. pos flags
        4. (For each gt_bboxes) Compute ignore flags based on adjacent anchor
           level then assign -1 to anchors w.r.t. ignore flags
        5. Assign anchor outside of image to -1

        Args:
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the ``InstanceData``
                in other places.
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                and ``labels``, with shape (k, ).
            img_meta (dict): Meta info of image.
            featmap_sizes (list[tuple[int, int]]): Feature map size each level.
            num_level_anchors (list[int]): The number of anchors in each level.
            anchor_scale (int): Scale of the anchor.
            anchor_strides (list[int]): Stride of the anchor.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes``
                attribute data that is ignored during training and testing.
                Defaults to None.
            allowed_border (int, optional): The border to allow the valid
                anchor. Defaults to 0.

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        if gt_instances_ignore is not None:
            raise NotImplementedError

        num_gts = len(gt_instances)
        num_bboxes = len(pred_instances)

        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        flat_anchors = pred_instances.priors
        flat_valid_flags = pred_instances.valid_flags
        mlvl_anchors = torch.split(flat_anchors, num_level_anchors)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = gt_bboxes.new_zeros((num_bboxes, ))
            assigned_gt_inds = gt_bboxes.new_zeros((num_bboxes, ),
                                                   dtype=torch.long)
            assigned_labels = gt_bboxes.new_full((num_bboxes, ),
                                                 -1,
                                                 dtype=torch.long)
            return AssignResult(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=max_overlaps,
                labels=assigned_labels)

        num_lvls = len(mlvl_anchors)
        r1 = (1 - self.center_ratio) / 2
        r2 = (1 - self.ignore_ratio) / 2

        scale = torch.sqrt((gt_bboxes[:, 2] - gt_bboxes[:, 0]) *
                           (gt_bboxes[:, 3] - gt_bboxes[:, 1]))
        min_anchor_size = scale.new_full(
            (1, ), float(anchor_scale * anchor_strides[0]))
        target_lvls = torch.floor(
            torch.log2(scale) - torch.log2(min_anchor_size) + 0.5)
        target_lvls = target_lvls.clamp(min=0, max=num_lvls - 1).long()

        # 1. assign 0 (negative) by default
        mlvl_assigned_gt_inds = []
        mlvl_ignore_flags = []
        for lvl in range(num_lvls):
            assigned_gt_inds = gt_bboxes.new_full((num_level_anchors[lvl], ),
                                                  0,
                                                  dtype=torch.long)
            ignore_flags = torch.zeros_like(assigned_gt_inds)
            mlvl_assigned_gt_inds.append(assigned_gt_inds)
            mlvl_ignore_flags.append(ignore_flags)

        for gt_id in range(num_gts):
            lvl = target_lvls[gt_id].item()
            featmap_size = featmap_sizes[lvl]
            stride = anchor_strides[lvl]
            anchors = mlvl_anchors[lvl]
            gt_bbox = gt_bboxes[gt_id, :4]

            # Compute regions
            ignore_region = calc_region(gt_bbox, r2, stride, featmap_size)
            ctr_region = calc_region(gt_bbox, r1, stride, featmap_size)

            # 2. Assign -1 to ignore flags
            ignore_flags = anchor_ctr_inside_region_flags(
                anchors, stride, ignore_region)
            mlvl_assigned_gt_inds[lvl][ignore_flags] = -1

            # 3. Assign gt_bboxes to pos flags
            pos_flags = anchor_ctr_inside_region_flags(anchors, stride,
                                                       ctr_region)
            mlvl_assigned_gt_inds[lvl][pos_flags] = gt_id + 1

            # 4. Assign -1 to ignore adjacent lvl
            if lvl > 0:
                d_lvl = lvl - 1
                d_anchors = mlvl_anchors[d_lvl]
                d_featmap_size = featmap_sizes[d_lvl]
                d_stride = anchor_strides[d_lvl]
                d_ignore_region = calc_region(gt_bbox, r2, d_stride,
                                              d_featmap_size)
                ignore_flags = anchor_ctr_inside_region_flags(
                    d_anchors, d_stride, d_ignore_region)
                mlvl_ignore_flags[d_lvl][ignore_flags] = 1
            if lvl < num_lvls - 1:
                u_lvl = lvl + 1
                u_anchors = mlvl_anchors[u_lvl]
                u_featmap_size = featmap_sizes[u_lvl]
                u_stride = anchor_strides[u_lvl]
                u_ignore_region = calc_region(gt_bbox, r2, u_stride,
                                              u_featmap_size)
                ignore_flags = anchor_ctr_inside_region_flags(
                    u_anchors, u_stride, u_ignore_region)
                mlvl_ignore_flags[u_lvl][ignore_flags] = 1

        # 4. (cont.) Assign -1 to ignore adjacent lvl
        for lvl in range(num_lvls):
            ignore_flags = mlvl_ignore_flags[lvl]
            mlvl_assigned_gt_inds[lvl][ignore_flags == 1] = -1

        # 5. Assign -1 to anchor outside of image
        flat_assigned_gt_inds = torch.cat(mlvl_assigned_gt_inds)
        assert (flat_assigned_gt_inds.shape[0] == flat_anchors.shape[0] ==
                flat_valid_flags.shape[0])
        inside_flags = anchor_inside_flags(flat_anchors, flat_valid_flags,
                                           img_meta['img_shape'],
                                           allowed_border)
        outside_flags = ~inside_flags
        flat_assigned_gt_inds[outside_flags] = -1

        assigned_labels = torch.zeros_like(flat_assigned_gt_inds)
        pos_flags = flat_assigned_gt_inds > 0
        assigned_labels[pos_flags] = gt_labels[flat_assigned_gt_inds[pos_flags]
                                               - 1]

        return AssignResult(
            num_gts=num_gts,
            gt_inds=flat_assigned_gt_inds,
            max_overlaps=None,
            labels=assigned_labels)
