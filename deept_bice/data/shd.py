
from typing import Callable, Optional

import torch
import numpy as np
from spikingjelly.datasets.shd import SpikingHeidelbergDigits

from deept.utils.globals import Context
from deept.data.dataset import register_dataset
from deept.data.dataloader import register_dataloader
from deept_bice.components.spikoder import RandomFixedSpecialTokenEncoder


torch.set_printoptions(edgeitems=100)

def pad_sequence_collate_as_dict(batch):
    inp = []
    tgt = []
    lens = []
    mask = []
    label_mask = []
    labels = []
    sos = []

    placeholder_inp = Context['placeholder_inp']
    placeholder_tgt = Context['placeholder_tgt']
    encoding_length = Context['encoding_length']

    for data_b, label_b, sos_b in batch:
        l = len(data_b)

        inp_b = np.append(data_b, placeholder_inp, axis=0)
        tgt_b = np.append(data_b, placeholder_tgt, axis=0)

        label_mask_b = torch.cat(
            [torch.zeros(l+1), torch.ones(encoding_length+1)],
            dim=0
        )
        
        inp.append(torch.as_tensor(inp_b))
        tgt.append(torch.as_tensor(tgt_b))

        mask.append(torch.ones(l+encoding_length+2))
        label_mask.append(label_mask_b)
        lens.append(l)
        labels.append(label_b)
        sos.append(sos_b)

    inp = torch.nn.utils.rnn.pad_sequence(inp, batch_first=True)
    tgt = torch.nn.utils.rnn.pad_sequence(tgt, batch_first=True)
    
    mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True)
    label_mask = torch.nn.utils.rnn.pad_sequence(label_mask, batch_first=True)

    labels = torch.as_tensor(labels)
    lens = torch.as_tensor(lens)
    sos = torch.cat(sos, dim=0)

    return {
        'tensors': {
            'inp': inp.float(),
            'tgt': tgt.float(),
            'labels': labels.long(),
            'lens': lens,
            'mask': mask.float(),
            'label_mask': label_mask.float(),
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
        J = config['input_dim'] // config['num_bins']
        T_l = config['encoding_length']
        
        Js = [0 for _ in range(J)]

        placeholder_inp = [Js for _ in range(T_l+1)]
        placeholder_tgt = [Js for _ in range(T_l+2)]
        
        Context.add_context('placeholder_inp', np.array(placeholder_inp))
        Context.add_context('placeholder_tgt', np.array(placeholder_tgt))
        Context.add_context('encoding_length', T_l)
        
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
