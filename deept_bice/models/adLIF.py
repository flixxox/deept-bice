import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import surrogate

from deept.utils.debug import my_print
from deept.components.model import register_model


@register_model('adLIF')
class adLIFSNN(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.input_dim = self.input_dim // self.num_bins

        self.__init_layers()
        self.__create_param_groups()

    @staticmethod
    def create_from_config(config):
        return adLIFSNN(
            input_keys=config['model_input'],
            input_dim=config['input_dim'],
            batch_size=config['batch_size'],
            hidden_size=config['hidden_size'],
            output_dim=config['output_dim'],
            dropout=config['dropout'],
            threshold=config['threshold'],
            n_layers=config['n_layers'],
            num_bins=config['num_bins'],
            cell_type=config['cell_type'],
        )

    def __init_layers(self):
        self.lif_nodes = nn.ModuleList([])
        
        input_dim = self.input_dim
        for i in range(1, self.n_layers+1):
            self.lif_nodes.append(
                adLIFLayer(
                    readout=False,
                    input_dim=input_dim,
                    hidden_size=self.hidden_size,
                    dropout=self.dropout,
                    threshold=self.threshold,
                    cell_type=self.cell_type
                )
            )
            input_dim = self.hidden_size

        self.readout = adLIFLayer(
            readout=True,
            input_dim=self.hidden_size,
            hidden_size=self.output_dim,
            dropout=self.dropout,
            threshold=self.threshold,
            cell_type=self.cell_type
        )

    def forward(self, x):
        x = x.permute(1,0,2) # [B, T, J] -> [T, B, J]

        all_spikes = []
        for i, snn_lay in enumerate(self.lif_nodes):
            x = snn_lay(x)
            all_spikes.append(x.permute(1,0,2).detach())

        x = self.readout(x)

        x = F.softmax(x, dim=2)
        x = x.sum(0) # [T, B, I] -> [B, I]
        
        return x, {
            'all_spikes': all_spikes
        }
    
    def __create_param_groups(self):
        weights = []
        weights_names = []
        norms = []
        norm_names = []

        for lif_node in self.lif_nodes + [self.readout]:
            for name, param in lif_node.named_parameters():
                if param.requires_grad:
                    if name == 'W':
                        weights.append(param)
                        weights_names.append(name)
                    else:
                        norms.append(param)
                        norm_names.append(name)

        my_print('~~ Param group: weights')
        my_print(weights_names)

        my_print('~~ Param group: norms')
        my_print(norm_names)

        self.param_groups = {
            'weights': [p for p in weights],
            'norms': [p for p in norms],
        }


class adLIFLayer(nn.Module):

    def __init__(self,
        **kwargs
    ):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        self.beta_lim = [np.exp(-1 / 30), np.exp(-1 / 120)]
        self.a_lim = [-1.0, 1.0]
        self.b_lim = [0.0, 2.0]
        self.spike_fct = surrogate.ATan(alpha=5.0)

        self.W = nn.Parameter(torch.empty((self.hidden_size, self.input_dim)))
        self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
        self.beta = nn.Parameter(torch.Tensor(self.hidden_size))
        self.a = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b = nn.Parameter(torch.Tensor(self.hidden_size))

        self.norm = nn.BatchNorm1d(self.hidden_size)

        if not self.readout:
            self.drop = nn.Dropout(self.dropout)

        if self.readout:
            self.cell_fn = self._readout_cell
        elif self.cell_type == 'soft':
            self.cell_fn = self._soft_reset_cell
        elif self.cell_type == 'hard':
            self.cell_fn = self._hard_reset_cell
        else:
            raise ValueError(f'Unrecognized cell_type argument {self.cell_type}!')

        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
        nn.init.uniform_(self.beta, self.beta_lim[0], self.beta_lim[1])
        nn.init.uniform_(self.a, self.a_lim[0], self.a_lim[1])
        nn.init.uniform_(self.b, self.b_lim[0], self.b_lim[1])

        self.__init()
    
    def __init(self):
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
        nn.init.uniform_(self.beta, self.beta_lim[0], self.beta_lim[1])
        nn.init.uniform_(self.a, self.a_lim[0], self.a_lim[1])
        nn.init.uniform_(self.b, self.b_lim[0], self.b_lim[1])

    def forward(self, x):
        B = x.shape[1]
        T = x.shape[0]
        J = x.shape[2]
        I = self.hidden_size
        device = x.device

        assert list(x.shape) == [T, B, J]

        alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])
        beta = torch.clamp(self.beta, min=self.beta_lim[0], max=self.beta_lim[1])
        a = torch.clamp(self.a, min=self.a_lim[0], max=self.a_lim[1])
        b = torch.clamp(self.b, min=self.b_lim[0], max=self.b_lim[1])

        U0 = torch.zeros(B, I).to(device)
        W0 = torch.zeros(B, I).to(device)

        x = F.linear(x, weight=self.W)

        assert list(x.shape) == [T, B, I]

        x = self.norm(x.reshape(T*B, I, 1)).reshape(T, B, I)
        
        assert list(x.shape) == [T, B, I]

        o = self.cell_fn(x, U0, W0, U0, alpha, beta, a, b)

        if not self.readout:
            o = self.drop(o)

        assert list(o.shape) == [T, B, I]

        return o

    def _soft_reset_cell(self, x, Ut, Wt, U0, alpha, beta, a, b):
        o = []
        B = x.shape[1]
        T = x.shape[0]
        I = x.shape[2]
        device = x.device

        theta = self.threshold
        St = torch.zeros(B, I).to(device)

        for t in range(T):
            Wt = alpha * Wt + a * Ut + b * St
            Ut = beta * (Ut - theta*St) + (1 - beta) * (x[t,:,:] - Wt)
            St = self.spike_fct(Ut - theta)
            o.append(St)

        return torch.stack(o, dim=0)

    def _hard_reset_cell(self, x, Ut, Wt, U0, alpha, beta, a, b):
        o = []
        B = x.shape[1]
        T = x.shape[0]
        I = x.shape[2]
        device = x.device

        theta = self.threshold
        St = torch.zeros(B, I).to(device)

        for t in range(T):
            Wt = alpha * Wt + a * Ut + b * St
            Ut = beta * Ut + (1 - beta) * (x[t,:,:] - Wt)
            St = self.spike_fct(Ut - theta)
            Ut = (1 - St.detach()) * Ut + St.detach() * U0
            o.append(St)

        return torch.stack(o, dim=0)

    def _readout_cell(self, x, Ut, Wt, U0, alpha, beta, a, b):
        o = []
        T = x.shape[0]
        for t in range(T):
            Ut = beta * Ut + (1 - beta) * x[t,:,:]
            o.append(Ut)
        return torch.stack(o, dim=0)