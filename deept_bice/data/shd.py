
from typing import Callable, Optional

import torch
import numpy as np
from spikingjelly.datasets import pad_sequence_collate
from spikingjelly.datasets.shd import SpikingHeidelbergDigits

from deept.data.dataset import register_dataset
from deept.data.dataloader import register_dataloader


def pad_sequence_collate_as_dict(batch):
    batch = pad_sequence_collate(batch)
    return {
        'tensors': {
            'data': batch[0].float(),
            'targets': batch[1].long()
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

    @staticmethod
    def create_from_config(config, is_train):
        return BinnedSpikingHeidelbergDigits(
            config['dataset_root'],
            config['num_bins'],
            train=is_train,
            duration=config['time_step']
        )

    def __getitem__(self, i: int):
        frames = np.load(self.frames_path[i], allow_pickle=True)['frames'].astype(np.float32)
        label = self.frames_label[i]
        binned_len = frames.shape[1]//self.n_bins
        binned_frames = np.zeros((frames.shape[0], binned_len))
        for i in range(binned_len):
            binned_frames[:,i] = frames[:, self.n_bins*i : self.n_bins*(i+1)].sum(axis=1)
        
        return binned_frames, label


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
