# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Tuple, Union

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

# from ...engine.utils import ext_loader

# ext_module = ext_loader.load_ext('_ext',
#                                  ['roi_pool_forward', 'roi_pool_backward'])


class RoIPoolFunction(Function):

    @staticmethod
    def symbolic(g, input, rois, output_size, spatial_scale):
        return g.op(
            'MaxRoiPool',
            input,
            rois,
            pooled_shape_i=output_size,
            spatial_scale_f=spatial_scale)

    @staticmethod
    def forward(ctx: Any,
                input: torch.Tensor,
                rois: torch.Tensor,
                output_size: Union[int, tuple],
                spatial_scale: float = 1.0) -> torch.Tensor:
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.input_shape = input.size()

        assert rois.size(1) == 5, 'RoI must be (idx, x1, y1, x2, y2)!'

        output_shape = (rois.size(0), input.size(1), ctx.output_size[0],
                        ctx.output_size[1])
        output = input.new_zeros(output_shape)
        argmax = input.new_zeros(output_shape, dtype=torch.int)

        ext_module.roi_pool_forward(
            input,
            rois,
            output,
            argmax,
            pooled_height=ctx.output_size[0],
            pooled_width=ctx.output_size[1],
            spatial_scale=ctx.spatial_scale)

        ctx.save_for_backward(rois, argmax)
        return output

    @staticmethod
    @once_differentiable
    def backward(
            ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None, None]:
        rois, argmax = ctx.saved_tensors
        grad_input = grad_output.new_zeros(ctx.input_shape)

        ext_module.roi_pool_backward(
            grad_output,
            rois,
            argmax,
            grad_input,
            pooled_height=ctx.output_size[0],
            pooled_width=ctx.output_size[1],
            spatial_scale=ctx.spatial_scale)

        return grad_input, None, None, None


roi_pool = RoIPoolFunction.apply


class RoIPool(nn.Module):

    def __init__(self,
                 output_size: Union[int, tuple],
                 spatial_scale: float = 1.0):
        super().__init__()

        self.output_size = _pair(output_size)
        self.spatial_scale = float(spatial_scale)

    def forward(self, input: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        return roi_pool(input, rois, self.output_size, self.spatial_scale)

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(output_size={self.output_size}, '
        s += f'spatial_scale={self.spatial_scale})'
        return s
