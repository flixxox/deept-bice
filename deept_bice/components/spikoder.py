import math

import torch
import torch.nn as nn

def create_spikoder(
    spikder_descr, input_dim, output_dim, encoding_length, sample_labels, resample_every_step
):
    if spikder_descr == 'linear':
        return LinearSpikoder(
            input_dim, output_dim, encoding_length,
            sample_labels, resample_every_step
        )
    elif spikder_descr == 'sinusoidal':
        return SinusoidalSpikoder(
            input_dim, output_dim, encoding_length,
            sample_labels, resample_every_step
        )
    else:
        raise ValueError(f'Did not recognize spikoder {spikder_descr}!')


class RandomFixedSpecialTokenEncoder(nn.Module):

    def __init__(self,
        input_dim,
        special_token_fr,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.special_token_fr = special_token_fr

        self.sos = nn.Parameter(
            (torch.rand(self.input_dim) > self.special_token_fr).float(),
            requires_grad=False
        )

    def get_special_token(self):
        return self.sos.view(1, self.input_dim)


class LinearSpikoder(nn.Module):

    def __init__(self,
        input_dim,
        num_classes,
        encoding_length,
        sample_labels,
        resample_every_step
    ):
        super().__init__()

        self.C = num_classes
        self.input_dim = input_dim
        self.T_l = encoding_length
        self.sample_labels = sample_labels
        self.resample_every_step = resample_every_step

        self.dfr = (1/self.C - 1/self.C**2)
        self.frs = [round(self.dfr*i, 3) for i in range(1, self.C+1)]

        self.dummy_param = nn.Parameter(torch.empty(0), requires_grad=False)

        if not self.sample_labels or not self.resample_every_step:
            labels = self.create_all_labels()
            self.labels = nn.Parameter(
                labels, requires_grad=False
            )

    def forward(self, x, tgt, lens, c, sos):

        B = x.shape[0]
        T_l = self.T_l
        device = x.device
        J = self.input_dim

        labels = self.get_all_labels()
        c = torch.index_select(labels, 0, c)

        assert list(c.shape) == [B, T_l, J]

        for b in range(B):
            
            x[b,lens[b],:] = sos[b]
            x[b,lens[b]+1:lens[b]+T_l+1,:] = c[b].detach()

            tgt[b,lens[b],:] = sos[b]
            tgt[b,lens[b]+1:lens[b]+T_l+1,:] = c[b]
            tgt[b,lens[b]+T_l+1,:] = sos[b]

        sos = sos.unsqueeze(1)
        x = torch.cat([sos, x], dim=1)

        return x, tgt, labels

    def get_all_labels(self):
        if not self.sample_labels or not self.resample_every_step:
            return self.labels
        else:
            return self.create_all_labels()

    def create_all_labels(self):
        if self.sample_labels:
            return self.sample_all_labels()
        else:
            return self.compute_all_labels()

    def sample_all_labels(self):
        C = self.C
        J = self.input_dim
        T_l = self.T_l
        device = self.dummy_param.device

        labels = []
        for c in range(C):
            c_seq = (torch.rand(T_l, J).to(device) < self.frs[c]).float()
            c_seq = c_seq.unsqueeze(0)
            labels.append(c_seq)
        labels = torch.cat(labels, dim=0)

        assert list(labels.shape) == [C, T_l, J]

        return labels

    def compute_all_labels(self):
        C = self.C
        J = self.input_dim
        T_l = self.T_l

        labels = []
        for c in range(C):
            ones = int(self.frs[c]*J)
            zeros = J-ones
            c_seq = torch.cat(
                [torch.ones(T_l, ones), torch.zeros(T_l, zeros)], dim=1
            )
            c_seq = c_seq.unsqueeze(0)
            labels.append(c_seq)
        labels = torch.cat(labels, dim=0)

        assert list(labels.shape) == [C, T_l, J]

        return labels


class SinusoidalSpikoder(nn.Module):

    def __init__(self,
        input_dim,
        num_classes,
        encoding_length,
        sample_labels,
        resample_every_step
    ):
        super().__init__()

        self.C = num_classes
        self.input_dim = input_dim
        self.T_l = encoding_length
        self.sample_labels = sample_labels
        self.resample_every_step = resample_every_step

        self.dummy_param = nn.Parameter(torch.empty(0), requires_grad=False)

        if not self.sample_labels or not self.resample_every_step:
            labels = self.create_all_labels()
            self.labels = nn.Parameter(
                labels, requires_grad=False
            )

    def forward(self, x, tgt, lens, c, sos):

        B = x.shape[0]
        T_l = self.T_l
        device = x.device
        J = self.input_dim

        labels = self.get_all_labels()
        c = torch.index_select(labels, 0, c)

        assert list(c.shape) == [B, T_l, J]

        for b in range(B):
            
            x[b,lens[b],:] = sos[b]
            x[b,lens[b]+1:lens[b]+T_l+1,:] = c[b].detach()

            tgt[b,lens[b],:] = sos[b]
            tgt[b,lens[b]+1:lens[b]+T_l+1,:] = c[b]
            tgt[b,lens[b]+T_l+1,:] = sos[b]

        sos = sos.unsqueeze(1)
        x = torch.cat([sos, x], dim=1)

        return x, tgt, labels

    def get_all_labels(self):
        if not self.sample_labels or not self.resample_every_step:
            return self.labels
        else:
            return self.create_all_labels()

    def create_all_labels(self):
        if self.sample_labels:
            return self.sample_all_labels()
        else:
            return self.compute_all_labels()

    def sample_all_labels(self):
        C = self.C
        J = self.input_dim
        T_l = self.T_l
        device = self.dummy_param.device

        labels = []
        for c in range(C):
            sign = 1
            if c % 2 == 0:
                sign = -1
            fq = ((c+1)/(C*2))
            fq = 2*math.pi*fq

            c_seq = []
            for t in range(1, T_l+1):
                fr = sign * 0.5*math.sin(fq*t)
                fr += 0.5
                ct_seq = (torch.rand(J) < fr).float()
                ct_seq = ct_seq.unsqueeze(0)
                c_seq.append(ct_seq)

            c_seq = torch.cat(c_seq, dim=0)
            c_seq = c_seq.unsqueeze(0)
            labels.append(c_seq)

        labels = torch.cat(labels, dim=0)
        labels = labels.to(device)

        assert list(labels.shape) == [C, T_l, J]

        return labels

    def compute_all_labels(self):
        C = self.C
        J = self.input_dim
        T_l = self.T_l
        device = self.dummy_param.device

        labels = []
        for c in range(C):
            sign = 1
            if c % 2 == 0:
                sign = -1
            fq = ((c+1)/(C*2))
            fq = 2*math.pi*fq

            c_seq = []
            for t in range(1, T_l+1):
                fr = sign * 0.5*math.sin(fq*t)
                fr += 0.5

                ones = int(fr*J)
                zeros = J-ones

                ct_seq = torch.cat(
                    [torch.ones(ones), torch.zeros(zeros)], dim=0
                )
                ct_seq = ct_seq.unsqueeze(0)
                c_seq.append(ct_seq)

            c_seq = torch.cat(c_seq, dim=0)
            c_seq = c_seq.unsqueeze(0)
            labels.append(c_seq)

        labels = torch.cat(labels, dim=0)
        labels = labels.to(device)

        assert list(labels.shape) == [C, T_l, J]

        return labels

