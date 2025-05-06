# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Custom replacement for `torch.nn.functional.grid_sample` that
supports arbitrarily high order gradients between the input and output.
Only works on 2D images and assumes
`mode='bilinear'`, `padding_mode='zeros'`, `align_corners=False`."""

import torch
import socket
import torch.nn.functional as F
from pkg_resources import parse_version

# pylint: disable=redefined-builtin
# pylint: disable=arguments-differ
# pylint: disable=protected-access

#----------------------------------------------------------------------------

enabled = True  # Enable the custom op by setting this to true.
_use_pytorch_1_11_api = parse_version(torch.__version__) >= parse_version('1.11.0a') # Allow prerelease builds of 1.11

#----------------------------------------------------------------------------

def grid_sample(input, grid, mode=None):
    if _should_use_custom_op():
        return GridSamplerFuncNoGrad.apply(input, grid)
    return torch.nn.functional.grid_sample(input=input, grid=grid, mode='bilinear', padding_mode='border', align_corners=True)

#----------------------------------------------------------------------------

def _should_use_custom_op():
    return enabled


class GridSamplerFuncNoGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, grid):
        result = F.grid_sample(z, grid, align_corners=True, mode='bilinear', padding_mode='border')
        # ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        # result, = ctx.saved_tensors
        # return: grad_input, grad_grid
        B, C, H, W = grad_output.shape
        reshaped_tensor = grad_output.contiguous().reshape(B, C, H//3, 3, W//3, 3)
        averaged_tensor = reshaped_tensor.mean(dim=[3, 5])
        grad_input = averaged_tensor * 0.1
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(grad_input)
        return grad_input, None

#----------------------------------------------------------------------------

if __name__ == "__main__":
    op = torch._C._jit_get_operation('aten::grid_sampler_2d_backward')
