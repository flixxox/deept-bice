import torch
import torch.nn as nn


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


class RandomFixedSpikoder(nn.Module):

    def __init__(self,
        input_dim,
        num_classes,
        encoding_length,
    ):
        super().__init__()

        self.C = num_classes
        self.input_dim = input_dim
        self.T_l = encoding_length

        self.dfr = (1/self.C - 1/self.C**2)
        self.frs = nn.Parameter(
            torch.Tensor([round(self.dfr*i, 3) for i in range(1, self.C+1)]),
            requires_grad=False
        )

        labels = self.precompute_all_labels()
        self.labels = nn.Parameter(
            labels, requires_grad=False
        )

    def precompute_all_labels(self):
        C = self.C
        J = self.input_dim
        T_l = self.T_l

        labels = []
        for c in range(C):
            
            c_seq = []
            for t in range(T_l):
                fr = self.frs[c]
                ct_seq = (torch.rand(J) < fr).float()
                ct_seq = ct_seq.view(1, J)
                c_seq.append(ct_seq)
            c_seq = torch.cat(c_seq, dim=0)
            c_seq = c_seq.view(1, T_l, J)

            labels.append(c_seq)
        labels = torch.cat(labels, dim=0)

        assert list(labels.shape) == [C, T_l, J]

        return labels

    def forward(self, x, c, sos):

        B = x.shape[0]
        J = self.input_dim
        T_l = self.T_l
        device = x.device

        labels = self.labels
        c = torch.index_select(labels, 0, c)

        assert list(c.shape) == [B, T_l, J]

        sos = sos.unsqueeze(1)

        inp = torch.cat(
            [sos, x, sos, c],
            dim=1
        )

        tgt = torch.cat(
            [x, sos, c, sos],
            dim=1
        )

        return inp, tgt, labels