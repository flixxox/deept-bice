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

from deept.components.model import register_model
from deept_bice.models.delayLmLIF import DelayLMLIFSNN


@register_model('toyWrapper')
class ToyWrapper(nn.Module):

    def __init__(self, config, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.model = DelayLMLIFSNN(
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

        self.param_groups = self.model.param_groups

    @staticmethod
    def create_from_config(config):
        return ToyWrapper(
            config,
            input_keys=config['model_input'],
        )

    def forward(self, x):
        out = self.model.search_forward(x)
        return out, {}
    
    # ~~~~~ Callbacks and Callback helpers

    def train_step_end_callback(self, step):
        self.model.train_step_end_callback(step)
        
    def train_epoch_end_callback(self, last_epoch):
        self.model.train_epoch_end_callback(last_epoch)

    def test_start_callback(self):
        self.model.test_start_callback()

    def test_end_callback(self):
        self.model.test_end_callback()