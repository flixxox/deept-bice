
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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