
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpikeFunctionBoxcar(torch.autograd.Function):
    """
    Compute surrogate gradient of the spike step function using
    box-car function similar to DECOLLE, Kaiser et al. (2020).
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.gt(0).float()

    def backward(ctx, grad_spikes):
        (x,) = ctx.saved_tensors
        grad_x = grad_spikes.clone()
        grad_x[x <= -0.5] = 0
        grad_x[x > 0.5] = 0
        return grad_x

class Binner(nn.Module):

    def __init__(self,
        num_bins
    ):
        super().__init__()
        self.num_bins = num_bins

    def __call__(self, x):
        B = x.shape[0]
        T = x.shape[1]
        J = x.shape[2]
        Bin = self.num_bins
        with torch.no_grad():
            x = x.contiguous().view(B, T, J//Bin, Bin).sum(-1)
            return x