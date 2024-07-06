
from typing import Callable, Optional

import torch
import numpy as np
from spikingjelly.datasets.shd import SpikingHeidelbergDigits

from deept.data.dataset import register_dataset
from deept.data.dataloader import register_dataloader
from deept_bice.components.spikoder import RandomFixedSpecialTokenEncoder


def pad_sequence_collate_as_dict(batch):
    data_list = []
    data_len_list = []
    label_list = []
    sos_list = []

    for data, label, sos in batch:
        data_list.append(torch.as_tensor(data))
        data_len_list.append(data.shape[0])
        label_list.append(label)
        sos_list.append(sos)

    data = torch.nn.utils.rnn.pad_sequence(data_list, batch_first=True)
    labels = torch.as_tensor(label_list)
    data_len = torch.as_tensor(data_len_list)
    sos = torch.cat(sos_list, dim=0)

    return {
        'tensors': {
            'data': data.float(),
            'labels': labels.long(),
            'data_len': data_len,
            'sos': sos
        }
    }

def pad_sequence_collate_as_dict_train(batch):
    batch = pad_sequence_collate_as_dict(batch)
    return [batch]


@register_dataset('shd')
class BinnedSpikingHeidelbergDigits(SpikingHeidelbergDigits):

    def __init__(
            self,
            root,
            n_bins,
            input_dim,
            special_token_fr,
            train=None,
            duration=None,
    ):
        super().__init__(
            root,
            train,
            'frame',
            None,
            None,
            duration,
        )
        self.n_bins = n_bins

        self.special_token_encoder = RandomFixedSpecialTokenEncoder(
            input_dim//n_bins,
            special_token_fr,
        )

    @staticmethod
    def create_from_config(config, is_train):
        return BinnedSpikingHeidelbergDigits(
            config['dataset_root'],
            config['num_bins'],
            config['input_dim'],
            config['special_token_fr'],
            train=is_train,
            duration=config['time_step'],
        )

    def __getitem__(self, i: int):
        frames = np.load(self.frames_path[i], allow_pickle=True)['frames'].astype(np.float32)
        label = self.frames_label[i]
        binned_len = frames.shape[1]//self.n_bins
        binned_frames = np.zeros((frames.shape[0], binned_len))
        for i in range(binned_len):
            binned_frames[:,i] = frames[:, self.n_bins*i : self.n_bins*(i+1)].sum(axis=1)

        sos = self.special_token_encoder.get_special_token()
        
        return binned_frames, label, sos


@register_dataloader('shd')
class SHDDataLoader:

    @staticmethod
    def create_from_config(config, dataset, is_train, num_workers):
        
        from torch.utils.data import DataLoader

        if is_train:
            collate_fn = pad_sequence_collate_as_dict_train
        else:
            collate_fn = pad_sequence_collate_as_dict
        
        return DataLoader(
            dataset,
            shuffle=is_train,
            num_workers=num_workers,
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
        )
