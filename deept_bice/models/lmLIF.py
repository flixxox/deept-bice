import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import surrogate

from deept.utils.debug import my_print
from deept.components.model import register_model
from deept_bice.components.spikoder import RandomFixedSpikoder


@register_model('lmLIF')
class LMLIFSNN(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.input_dim = self.input_dim // self.num_bins

        self.spikoder = RandomFixedSpikoder(
            self.input_dim,
            self.output_dim,
            self.encoding_length
        )
        
        input_dim = self.input_dim
        self.lif_nodes = nn.ModuleList([])
        for i in range(1, self.n_layers):
            self.lif_nodes.append(
                LMLIFLayer(
                    readout=False,
                    input_dim=input_dim,
                    hidden_size=self.hidden_size,
                    dropout=self.dropout,
                    threshold=self.threshold,
                    beta_init_min=self.beta_init_min,
                    beta_init_max=self.beta_init_max,
                    cell_type=self.cell_type
                )
            )
            input_dim = self.hidden_size

        self.lif_nodes.append(
            LMLIFLayer(
                readout=True,
                input_dim=self.hidden_size,
                hidden_size=self.input_dim,
                dropout=self.dropout,
                threshold=self.threshold,
                beta_init_min=self.beta_init_min,
                beta_init_max=self.beta_init_max,
                cell_type=self.cell_type
            )
        )

        self.__create_param_groups()

    @staticmethod
    def create_from_config(config):
        return LMLIFSNN(
            input_keys=config['model_input'],
            input_dim=config['input_dim'],
            batch_size=config['batch_size'],
            hidden_size=config['hidden_size'],
            output_dim=config['output_dim'],
            dropout=config['dropout'],
            threshold=config['threshold'],
            n_layers=config['n_layers'],
            num_bins=config['num_bins'],
            beta_init_min=config['beta_init_min'],
            beta_init_max=config['beta_init_max'],
            cell_type=config['cell_type'],
            encoding_length=config['encoding_length'],
            loss_function_descr=config['criterions'][0]['loss_function']
        )

    def forward(self, x, c, sos):
        B = x.shape[0]
        J = x.shape[2]
        C = self.output_dim
        T_d = x.shape[1]
        T_l = self.encoding_length

        assert list(sos.shape) == [B, J]

        x, tgt, labels = self.spikoder(x, c, sos)

        assert list(x.shape) == [B, T_d+T_l+2, J]
        assert list(labels.shape) == [C, T_l, J]

        for snn_lay in self.lif_nodes:
            x = snn_lay(x)
        
        return x, {'tgt': tgt, 'label_seqs': labels}

    def __create_param_groups(self):
        weights = []
        weights_names = []
        norms = []
        norm_names = []

        for lif_node in self.lif_nodes:
            for name, param in lif_node.named_parameters():
                if param.requires_grad:
                    if 'norm.' in name:
                        norms.append(param)
                        norm_names.append(name)
                    elif 'beta' in name:
                        norms.append(param)
                        norm_names.append(name)
                    else:
                        weights.append(param)
                        weights_names.append(name)

        my_print('~~ Param group: weights')
        my_print(weights_names)

        my_print('~~ Param group: norms')
        my_print(norm_names)

        self.param_groups = {
            'weights': [p for p in weights],
            'norms': [p for p in norms],
        }


class LMLIFLayer(nn.Module):

    def __init__(self,
        **kwargs
    ):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.spike_fct = surrogate.ATan(alpha=5.0)

        self.W = nn.Parameter(
            torch.empty((self.hidden_size, self.input_dim))
        )

        self.beta = nn.Parameter(torch.Tensor(self.hidden_size))
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.BatchNorm1d(self.hidden_size)
    
        if self.cell_type == 'soft':
            self.cell_fn = self._soft_reset_cell
        elif self.cell_type == 'hard':
            self.cell_fn = self._hard_reset_cell
        else:
            raise ValueError(f'Unrecognized cell_type argument {self.cell_type}!')

        if not self.readout:
            self.drop = nn.Dropout(self.dropout)

        self.__init()
    
    def __init(self):
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        nn.init.uniform_(self.beta, self.beta_init_min, self.beta_init_max)

    def forward(self, x):
        B = x.shape[1]
        T = x.shape[0]
        J = x.shape[2]
        I = self.hidden_size
        device = x.device

        assert list(x.shape) == [T, B, J]

        x = F.linear(x, weight=self.W)
        beta = self.sigmoid(self.beta)
        U0 = torch.zeros(B, I).to(device)

        assert list(x.shape) == [T, B, I]

        x = self.norm(x.reshape(T*B, I, 1)).reshape(T, B, I)
        
        assert list(x.shape) == [T, B, I]

        o = self.cell_fn(x, U0, U0, beta)

        if not self.readout:
            o = self.drop(o)

        assert list(o.shape) == [T, B, I]

        return o

    def _soft_reset_cell(self, x, Ut, U0, beta):
        o = []
        B = x.shape[1]
        T = x.shape[0]
        I = x.shape[2]
        device = x.device

        theta = self.threshold
        St = torch.zeros(B, I).to(device)

        for t in range(T):
            Ut = beta * (Ut - theta*St) + (1 - beta) * x[t,:,:]
            St = self.spike_fct(Ut - theta)
            o.append(St)

        return torch.stack(o, dim=0)

    def _hard_reset_cell(self, x, Ut, U0, beta):
        o = []
        T = x.shape[0]
        theta = self.threshold

        for t in range(T):
            Ut = beta * Ut + (1 - beta) * x[t,:,:]
            St = self.spike_fct(Ut - theta)
            Ut = (1 - St.detach()) * Ut + St.detach() * U0
            o.append(St)

        return torch.stack(o, dim=0)

    def _readout_cell(self, x, Ut, U0, beta):
        o = []
        T = x.shape[0]
        for t in range(T):
            Ut = beta * Ut + (1 - beta) * x[t,:,:]
            o.append(Ut)
        return torch.stack(o, dim=0)
