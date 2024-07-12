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
from deept_bice.components.spikoder import (
    create_spikoder,
    RandomFixedSpecialTokenEncoder
)



@register_model('delayLmLIF')
class DelayLMLIFSNN(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.input_dim = self.input_dim // self.num_bins

        self.spikoder = create_spikoder(
            self.spikoder_descr,
            self.input_dim,
            self.output_dim,
            self.encoding_length,
            self.sample_labels,
            self.resample_every_step
        )

        self.special_token_encoder = RandomFixedSpecialTokenEncoder(
            self.input_dim,
            self.special_token_fr,
        )
        
        input_dim = self.input_dim
        self.lif_nodes = nn.ModuleList([])
        for i in range(1, self.n_layers):
            self.lif_nodes.append(
                DelayLMLIFLayer(
                    layer_num=i,
                    readout=False,
                    use_norm=True,
                    input_dim=input_dim,
                    hidden_size=self.hidden_size,
                    dropout=self.dropout,
                    cell_type=self.cell_type,
                    threshold=self.threshold,
                    beta_init_min=self.beta_init_min,
                    beta_init_max=self.beta_init_max,
                    # delayLmLIF
                    tau=(self.init_tau+1e-9)/self.time_step,
                    max_delay=self.max_delay,
                    sigma_init=self.sigma_init, 
                )
            )
            input_dim = self.hidden_size

        self.lif_nodes.append(
            DelayLMLIFLayer(
                layer_num=self.n_layers,
                readout=True,
                use_norm=True,
                input_dim=self.hidden_size,
                hidden_size=self.input_dim,
                dropout=self.dropout,
                cell_type=self.cell_type,
                threshold=self.threshold,
                beta_init_min=self.beta_init_min,
                beta_init_max=self.beta_init_max,
                # delayLmLIF
                tau=(self.init_tau+1e-9)/self.time_step,
                max_delay=self.max_delay,
                sigma_init=self.sigma_init, 
            )
        )

        self.sigma = self.lif_nodes[-1].delay.SIG[0,0,0,0].detach().cpu().item()

        self.__create_param_groups()

    @staticmethod
    def create_from_config(config):
        return DelayLMLIFSNN(
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
            beta_init_min=config['beta_init_min'],
            beta_init_max=config['beta_init_max'],
            encoding_length=config['encoding_length'],
            spikoder_descr=config['spikoder'],
            sample_labels=config['sample_labels'],
            special_token_fr=config['special_token_fr'],
            resample_every_step=config['resample_every_step'],
            similarity_function_descr=config['similarity_function'],
            # delayLIF
            init_tau=config['init_tau'],
            time_step=config['time_step'],
            max_delay=config['max_delay'],
            sigma_init=config['sigma_init'],
            sigma_decrease_final_epoch=config['sigma_decrease_final_epoch']
        )

    def __create_param_groups(self):

        weights = []
        delays = []
        batchnorms = []
        delays_names = []
        weights_names = []
        batchnorm_names = []
        for lif_node in self.lif_nodes:
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

    def forward(self, x, tgt, lens, c):
        B = x.shape[0]
        T = x.shape[1]
        J = self.input_dim
        C = self.output_dim
        T_l = self.encoding_length

        sos = self.special_token_encoder()
        sos = sos.repeat(B, 1)

        assert list(x.shape) == [B, T, J]
        assert list(tgt.shape) == [B, T+1, J]
        assert list(lens.shape) == [B]
        assert list(c.shape) == [B]
        assert list(sos.shape) == [B, J]

        x, tgt, labels = self.spikoder(x, tgt, lens, c, sos)

        assert list(x.shape) == [B, T+1, J]
        assert list(tgt.shape) == [B, T+1, J]
        assert list(labels.shape) == [C, T_l, J]

        x = x.permute(1,0,2) # [B, T, J] -> [T, B, J]
        for snn_lay in self.lif_nodes:
            x = snn_lay(x)
        x = x.permute(1,0,2) # [T, B, J] -> [B, T, J]

        if torch.mean(x) < 0.001:
            print('Warning! Low activity in output!')
        
        return x, {'tgt': tgt, 'label_seqs': labels}

    def search_forward(self, x):
        x = x.permute(1,0,2) # [B, T, J] -> [T, B, J]
        for snn_lay in self.lif_nodes:
            x = snn_lay(x)
        x = x.permute(1,0,2) # [T, B, J] -> [B, T, J]
        return x

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


class DelayLMLIFLayer(nn.Module):

    def __init__(self,
        **kwargs
    ):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        assert isinstance(self.tau, float) and self.tau > 1.

        self.beta = nn.Parameter(torch.Tensor(self.hidden_size))
        self.sigmoid = nn.Sigmoid()

        self.init_pos_a = -self.max_delay // 2
        self.init_pos_b = self.max_delay // 2

        self.left_padding = self.max_delay - 1
        # In contrast to the original implementation we 
        # do not use right_padding to preserve the time dimension
        self.right_padding = 0 #(self.max_delay - 1) // 2

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
        
        self.random_batchwise_init = RandomBatchwiseInit(0, 1)

        if self.use_norm:
            self.norm = nn.BatchNorm1d(self.hidden_size)
        if not self.readout:
            self.drop = DropoutOverTime(self.dropout)

        if self.cell_type == 'soft':
            self.cell_fn = self._soft_reset_cell
        elif self.cell_type == 'hard':
            self.cell_fn = self._hard_reset_cell
        else:
            raise ValueError(f'Unrecognized cell_type argument {self.cell_type}!')

        self.__init()
    
    def __init(self):

        torch.nn.init.kaiming_uniform_(self.delay.weight, nonlinearity='relu')
        nn.init.uniform_(self.beta, self.beta_init_min, self.beta_init_max)

        torch.nn.init.uniform_(self.delay.P, a=self.init_pos_a, b=self.init_pos_b)
        self.delay.clamp_parameters()

        torch.nn.init.constant_(self.delay.SIG, self.sigma_init)
        self.delay.SIG.requires_grad = False

    def forward(self, x):
        T = x.shape[0]
        B = x.shape[1]
        J = x.shape[2]
        I = self.hidden_size
        device = x.device

        # ~~~~ Delay

        assert list(x.shape) == [T, B, J]

        x = x.permute(1,2,0) # [T, B, J] -> [B, J, T]
        x = F.pad(x, (self.left_padding, self.right_padding), 'constant', 0)
        x = self.delay(x)
        x = x.permute(2,0,1) # [B, I, T] -> [T, B, I]

        # ~~~~ Norm

        assert list(x.shape) == [T, B, I]

        if self.use_norm:
            x = self.norm(x.reshape(T*B, I, 1)).reshape(T, B, I)

        # ~~~~ LIF

        assert list(x.shape) == [T, B, I]

        U0 = self.random_batchwise_init(B, I)
        beta = self.sigmoid(self.beta)

        o = self.cell_fn(x, U0, U0, self.beta)

        if not self.readout:
            o = self.drop(o)

        assert list(o.shape) == [T, B, I]

        return o

    def _soft_reset_cell(self, x, Ut, U0, beta):
        o = []
        T = x.shape[0]
        B = x.shape[1]
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


class RandomBatchwiseInit(nn.Module):

    def __init__(
        self,
        init_min,
        init_max
    ):
        super().__init__()
        self.init_min = init_min
        self.init_max = init_max
        self.dummy_param = nn.Parameter(torch.empty(0), requires_grad=False)

    def forward(self, *shape):
        device = self.dummy_param.get_device()
        return (torch.FloatTensor(*shape)
            .uniform_(self.init_min, self.init_max)
            .to(device)
        )