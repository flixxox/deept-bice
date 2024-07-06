import math
from os import remove
from os.path import join
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from DCLS.construct.modules import Dcls1d
from spikingjelly.activation_based import surrogate

from deept.utils.debug import my_print
from deept.utils.globals import Settings
from deept.components.model import register_model
from deept_bice.components.modules import DropoutOverTime


@register_model('delayLIF')
class DelayLIFSNN(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.input_dim = self.input_dim // self.num_bins

        self.__init_layers()
        self.__create_param_groups()

        self.sigma = self.lif_nodes[-1].delay.SIG[0,0,0,0].detach().cpu().item()

    @staticmethod
    def create_from_config(config):
        return DelayLIFSNN(
            input_keys=config['model_input'],
            input_dim=config['input_dim'],
            batch_size=config['batch_size'],
            hidden_size=config['hidden_size'],
            output_dim=config['output_dim'],
            dropout=config['dropout'],
            threshold=config['threshold'],
            n_layers=config['n_layers'],
            num_bins=config['num_bins'],
            # delayLIF
            init_tau=config['init_tau'],
            time_step=config['time_step'],
            max_delay=config['max_delay'],
            sigma_init=config['sigma_init'],
            sigma_decrease_final_epoch=config['sigma_decrease_final_epoch']
        )

    def __init_layers(self):

        self.lif_nodes = nn.ModuleList([])
        
        input_dim = self.input_dim
        for i in range(1, self.n_layers+1):
            self.lif_nodes.append(
                DelayLIFLayer(
                    readout=False,
                    use_norm=True,
                    input_dim=input_dim,
                    hidden_size=self.hidden_size,
                    dropout=self.dropout,
                    threshold=self.threshold,
                    # delayLIF
                    tau=(self.init_tau+1e-9)/self.time_step,
                    max_delay=self.max_delay,
                    sigma_init=self.sigma_init, 
                )
            )
            input_dim = self.hidden_size

        self.readout = DelayLIFLayer(
            readout=True,
            use_norm=False,
            input_dim=self.hidden_size,
            hidden_size=self.output_dim,
            dropout=self.dropout,
            threshold=1e9,
            # delayLIF
            tau=(self.init_tau+1e-9)/self.time_step,
            max_delay=self.max_delay,
            sigma_init=self.sigma_init, 
        )

    def __create_param_groups(self):

        weights = []
        delays = []
        batchnorms = []
        delays_names = []
        weights_names = []
        batchnorm_names = []
        for lif_node in self.lif_nodes + [self.readout]:
            for name, param in lif_node.named_parameters():
                if param.requires_grad:
                    if name == 'delay.P':
                        delays.append(param)
                        delays_names.append(name)
                    elif 'norm.' in name:
                        batchnorms.append(param)
                        batchnorm_names.append(name)
                    else:
                        weights.append(param)
                        weights_names.append(name)

        my_print('~~ Param group: weights')
        my_print(weights_names)

        my_print('~~ Param group: delays')
        my_print(delays_names)

        my_print('~~ Param group: batchnorms')
        my_print(batchnorm_names)

        self.param_groups = {
            'weights': [p for p in weights],
            'delays': [p for p in delays],
            'batchnorms': [p for p in batchnorms],
        }

    def forward(self, x):

        x = x.permute(1,0,2)

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

    # ~~~~~ Callbacks and Callback helpers

    def train_step_end_callback(self, step):
        for lif_node in self.lif_nodes:
            lif_node.delay.clamp_parameters()
        
    def train_epoch_end_callback(self, last_epoch):
        self.__decrease_sig(last_epoch)

    def test_start_callback(self):
        self.__backup()
        for lif_node in self.lif_nodes:
            lif_node.delay.SIG *= 0
            lif_node.delay.version = 'max'
            lif_node.delay.DCK.version = 'max'
        self.__round_delay()

    def test_end_callback(self):
        my_print('Loading backup!')
        for lif_node in self.lif_nodes:
            lif_node.delay.version = 'gauss'
            lif_node.delay.DCK.version = 'gauss'
        self.load_state_dict(torch.load(self.backup_checkpoint), strict=True)
        remove(self.backup_checkpoint)

    # ~~~~~ Helpers

    def __decrease_sig(self, last_epoch):

        if last_epoch < self.sigma_decrease_final_epoch and self.sigma > 0.23:
            alpha = (0.23 / self.sigma_init) ** (1 / self.sigma_decrease_final_epoch)
            sigma = self.sigma * alpha

            my_print(f'Updating sigma from {self.sigma:4.2f} to {sigma:4.2f}!')
            self.sigma = sigma

            for lif_node in self.lif_nodes:
                lif_node.delay.SIG *= alpha

    def __backup(self):
        my_print('Creating backup!')
        self.backup_checkpoint = join(Settings.get_dir('checkpoint_dir'), 'tmp_backup.pt')
        torch.save(self.state_dict(), self.backup_checkpoint)

    def __round_delay(self):
        with torch.no_grad():
            for lif_node in self.lif_nodes:
                lif_node.delay.P.round_()
                lif_node.delay.clamp_parameters()

    def __load_from_delay_snn_init(self):
        ckpt = {}
        ckpt_loaded = torch.load('/Users/fschmidt/code/SNN-delays/snn_delays_init.pt')
        for k,v in ckpt_loaded.items():
            k_new = k.replace('model.0', 'lif_nodes.0.delay')
            k_new = k_new.replace('model.1', 'lif_nodes.0.norm')
            k_new = k_new.replace('model.4', 'lif_nodes.1.delay')
            k_new = k_new.replace('model.5', 'lif_nodes.1.norm')
            k_new = k_new.replace('model.8', 'readout.delay')
            ckpt[k_new] = ckpt_loaded[k]
        self.load_state_dict(ckpt, strict=True)


class DelayLIFLayer(nn.Module):

    def __init__(self,
        **kwargs
    ):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        assert isinstance(self.tau, float) and self.tau > 1.

        self.beta = (1. - 1. / self.tau)

        self.init_pos_a = -self.max_delay // 2
        self.init_pos_b = self.max_delay // 2

        self.left_padding = self.max_delay - 1
        self.right_padding = (self.max_delay - 1) // 2

        self.spike_fct = surrogate.ATan(alpha=5.0)

        self.delay = Dcls1d(
            self.input_dim,
            self.hidden_size,
            groups=1,
            bias=False,
            version='gauss',
            kernel_count=1,
            dilated_kernel_size=self.max_delay,
        )
        
        if self.use_norm:
            self.norm = nn.BatchNorm1d(self.hidden_size)
        if not self.readout:
            self.drop = DropoutOverTime(self.dropout)

        self.__init()
    
    def __init(self):

        torch.nn.init.kaiming_uniform_(self.delay.weight, nonlinearity='relu')
        
        torch.nn.init.uniform_(self.delay.P, a=self.init_pos_a, b=self.init_pos_b)
        self.delay.clamp_parameters()

        torch.nn.init.constant_(self.delay.SIG, self.sigma_init)
        self.delay.SIG.requires_grad = False

    def forward(self, x):

        B = x.shape[1]
        T = x.shape[0]
        J = x.shape[2]
        I = self.hidden_size
        device = x.device

        # ~~~~ Delay

        assert list(x.shape) == [T, B, J]

        x = x.permute(1,2,0) # [T, B, J] -> [B, J, T]
        x = F.pad(x, (self.left_padding, self.right_padding), 'constant', 0)
        x = self.delay(x)
        x = x.permute(2,0,1) # [B, I, T] -> [T, B, I]

        T = x.shape[0] # T changed during conv

        # ~~~~ Norm

        assert list(x.shape) == [T, B, I]

        if self.use_norm:
            x = self.norm(x.reshape(T*B, I, 1)).reshape(T, B, I)

        # ~~~~ LIF

        assert list(x.shape) == [T, B, I]

        Ut = torch.zeros(B, I).to(device)

        o = []
        for t in range(T):
            Ut = Ut * self.beta + x[t,:,:]

            if self.readout:
                o.append(Ut)
            else:
                St = self.spike_fct(Ut - self.threshold)
                Ut = (1 - St.detach()) * Ut
                o.append(St)

        o = torch.stack(o, dim=0)

        if not self.readout:
            o = self.drop(o)

        assert list(o.shape) == [T, B, I]

        return o