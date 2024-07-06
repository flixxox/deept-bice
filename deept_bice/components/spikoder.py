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


class Spikoder(nn.Module):

    def __init__(self,
        **kwargs
    ):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

    def forward(self, x, c, sos):

        sos = sos.unsqueeze(1)
        c = self.encode_label(c)

        inp = torch.cat(
            [sos, x, sos, c],
            dim=1
        )

        tgt = torch.cat(
            [x, sos, c, sos],
            dim=1
        )

        return inp, tgt, label_seq

    def encode_label(self, c):
        pass