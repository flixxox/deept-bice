
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


class DropoutOverTime(nn.Module):

    def __init__(self, p, time_dim=0):
        super().__init__()
        self.p = p
        self.time_dim = time_dim
        self.sample_idx = nn.Parameter(torch.zeros(1).to(torch.int), requires_grad=False)
    
    def forward(self, x):
        T = x.shape[self.time_dim]
        sample = x.index_select(self.time_dim, self.sample_idx)
        mask = torch.ones_like(sample.data)
        mask = F.dropout(mask, self.p, training=self.training)
        if self.time_dim == 0:
            mask = mask.repeat(T, 1, 1)
        elif self.time_dim == 1:
            mask = mask.repeat(1, T, 1)
        return x * mask