import torch
import numpy as np

# JIT
from torch.utils.cpp_extension import load
dvxlr = load("dvxlr", sources=["lib/dvxlr/dvxlr.cpp", "lib/dvxlr/dvxlr.cu"], verbose=True)


class DifferentiableVoxelRenderingLayer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sigma, origin, points, tindex):
        pred_dist, gt_dist, dd_dsigma, indices = dvxlr.render(sigma,
                                                              origin,
                                                              points,
                                                              tindex)
        ctx.save_for_backward(dd_dsigma, indices, tindex, sigma)
        return pred_dist, gt_dist

    @staticmethod
    def backward(ctx, gradpred, gradgt):
        dd_dsigma, indices, tindex, sigma_shape = ctx.saved_tensors
        elementwise_mult = gradpred[..., None] * dd_dsigma

        grad_sigma = dvxlr.get_grad_sigma(elementwise_mult, indices, tindex, sigma_shape)[0]

        return grad_sigma, None, None, None


DifferentiableVoxelRendering = DifferentiableVoxelRenderingLayer.apply
