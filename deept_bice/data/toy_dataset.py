
from typing import Callable, Optional

import torch
import numpy as np
from torch.utils.data import Dataset

from deept.utils.globals import Context
from deept.data.dataset import register_dataset
from deept.data.dataloader import register_dataloader
from deept_bice.components.spikoder import RandomFixedSpecialTokenEncoder


def pad_toy_sequence_collate_as_dict(batch):
    inp = []
    tgt = []
    mask = []
    label_mask = []

    for data_b, sos_b in batch:
        l = data_b.shape[0]

        assert isinstance(data_b, torch.Tensor)
        assert isinstance(sos_b, torch.Tensor)

        inp_b = torch.cat([sos_b, data_b], dim=0)
        tgt_b = torch.cat([data_b, sos_b], dim=0)
        
        inp.append(inp_b)
        tgt.append(tgt_b)

        mask.append(torch.ones(l+1))
        label_mask.append(torch.zeros(l+1))

    inp = torch.stack(inp, dim=0)
    tgt = torch.stack(tgt, dim=0)
    mask = torch.stack(mask, dim=0)
    label_mask = torch.stack(label_mask, dim=0)

    return {
        'tensors': {
            'inp': inp.float(),
            'tgt': tgt.float(),
            'mask': mask.float(),
            'label_mask': label_mask.float(),
        }
    }

def pad_toy_sequence_collate_as_dict_train(batch):
    batch = pad_toy_sequence_collate_as_dict(batch)
    return [batch]


@register_dataset('toy')
class ToyDataset(Dataset):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.input_dim = self.input_dim // self.num_bins

        if Context.has_context('toy_dataset'):
            self.dataset = Context['toy_dataset']
        else:
            self.dataset = self.create_toy_dataset()
            Context.add_context('toy_dataset', self.dataset)

        if Context.has_context('special_token_encoder'):
            self.special_token_encoder = Context['special_token_encoder']
        else:
            self.special_token_encoder = RandomFixedSpecialTokenEncoder(
                self.input_dim,
                self.special_token_fr,
            )
            Context.add_context('special_token_encoder', self.special_token_encoder)

    def create_toy_dataset(self):
        J = self.input_dim
        N = self.n_samples
        T = self.sample_length
        dataset = []
        for n in range(N):
            sample = torch.bernoulli(torch.rand(T, J))
            dataset.append(sample)
        return dataset

    @staticmethod
    def create_from_config(config, is_train):
        return ToyDataset(
            num_bins=config['num_bins'],
            input_dim=config['input_dim'],
            n_samples=config['n_samples'],
            sample_length=config['sample_length'],
            special_token_fr=config['special_token_fr']
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i: int):
        sos = self.special_token_encoder()
        return self.dataset[i], sos


@register_dataloader('toy')
class ToyDataLoader:

    @staticmethod
    def create_from_config(config, dataset, is_train, num_workers):
        
        from torch.utils.data import DataLoader

        if is_train:
            collate_fn = pad_toy_sequence_collate_as_dict_train
        else:
            collate_fn = pad_toy_sequence_collate_as_dict
        
        return DataLoader(
            dataset,
            shuffle=is_train,
            num_workers=num_workers,
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
        )
