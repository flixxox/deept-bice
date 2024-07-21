import torch
import torch.nn as nn
import torch.nn.functional as F

from deept_bice.components.scores import (
    _activity_similarity_fn,
    _per_neuron_similarity_fn
)


class GreedySearch(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.input_dim = self.input_dim // self.num_bins

        if self.similarity_fn_descr == 'neuron':
            self.similarity_fn = _per_neuron_similarity_fn
        elif self.similarity_fn_descr == 'activity':
            self.similarity_fn = _activity_similarity_fn
        else:
            raise ValueError(f'Did not recognize loss_function! Given: {self.similarity_fn_descr}')

    @staticmethod
    def create_from_config(config, model):
        return GreedySearch(
            model=model,
            num_bins=config['num_bins'],
            T_l = config['encoding_length'],
            input_dim = config['input_dim'],
            num_classes = config['output_dim'],
            similarity_fn_descr = config['similarity_function']
        )

    def forward(self, x, lens, early_abort_at=None):
        B = x.shape[0]
        T = x.shape[1]
        J = self.input_dim
        C = self.num_classes
        T_l = self.T_l
        device = x.device

        T_fin = T_l
        if early_abort_at is not None:
            T_fin = early_abort_at
        
        label_seqs = self.model.spikoder.get_all_labels()

        sos = self.model.special_token_encoder()
        sos = sos.repeat(B, 1)

        x = self.place_preceeding_and_following_sos(x, lens, sos)
        T = T + 1 # Because we placed sos at the beginning

        # Those are the timesteps at which we can 
        # find our prediction per batch entry. 
        idx = torch.arange(T_l).view(1, T_l).to(device)
        idx = idx + lens.unsqueeze(1).repeat(1, T_l) + 1
        idx = idx.unsqueeze(-1).repeat(1,1,J).long()

        assert list(idx.shape) == [B, T_l, J]

        # We do the first step here to initialize pred_label_seq.
        # pred_label_seq will contain the predicted label sequence.
        pred = self.model.search_forward(x)
        pred_label_seq = torch.gather(
            pred,
            1, 
            idx[:,0,:].unsqueeze(1)
        )

        assert list(label_seqs.shape) == [C, T_l, J]
        assert list(pred_label_seq.shape) == [B, 1, J]

        pred_label_sofar = self.get_most_similar_label_sofar(
            pred_label_seq, label_seqs, 1
        )

        t = 1
        while t <= T_fin:

            assert list(pred_label_sofar.shape) == [B]
            
            x = self.update_x_with_most_similar_label_sequence_sofar(
                x, pred_label_sofar, label_seqs, idx
            )    

            assert list(x.shape) == [B, T, J]

            pred = self.model.search_forward(x)
            pred_label_seq = torch.gather(
                pred, 1, idx[:,:t,:]
            )

            assert list(pred_label_seq.shape) == [B, t, J]

            pred_label_sofar = self.get_most_similar_label_sofar(
                pred_label_seq, label_seqs, t
            )

            t += 1

        assert list(pred_label_sofar.shape) == [B]
        assert list(pred_label_seq.shape) == [B, T_fin, J]

        return pred_label_sofar, pred_label_seq

    def place_preceeding_and_following_sos(self, x, lens, sos):
        # Todo: Ideally this is done before batching
        for b in range(x.shape[0]):
            x[b,lens[b],:] = sos[b]
        sos = sos.unsqueeze(1)
        x = torch.cat([sos, x], dim=1)
        return x

    def get_most_similar_label_sofar(self, pred_label_seq, label_seqs, t):
        B = pred_label_seq.shape[0]
        J = self.input_dim
        C = self.num_classes

        label_seqs = label_seqs[:,:t,:]
        
        assert list(label_seqs.shape) == [C, t, J]
        assert list(pred_label_seq.shape) == [B, t, J]

        label_seqs = label_seqs.unsqueeze(0)
        pred_label_seq = pred_label_seq.unsqueeze(1)

        similarity = self.similarity_fn(pred_label_seq, label_seqs)

        assert list(similarity.shape) == [B, C, t]

        similarity = torch.mean(similarity, dim=-1)
        predicted_label_sofar = torch.argmin(similarity, dim=-1)

        return predicted_label_sofar

    def update_x_with_most_similar_label_sequence_sofar(self, x, pred_label_sofar, label_seqs, idx):
        B = x.shape[0]
        T = x.shape[1]
        J = self.input_dim
        C = self.num_classes
        T_l = self.T_l

        pred_label_sofar = pred_label_sofar.view(B, 1, 1)
        pred_label_sofar = pred_label_sofar.repeat(1, T_l, J)
        label_seqs = torch.gather(label_seqs, 0, pred_label_sofar)

        idx = idx[:,:,0] # We dont need the repeated tensor

        assert list(idx.shape) == [B, T_l]
        assert list(label_seqs.shape) == [B, T_l, J]

        for b in range(B):
            for t in range(T_l):
                x[b, idx[b,t]] = label_seqs[b, t]

        return x


class ParallelSearch(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.input_dim = self.input_dim // self.num_bins

        if self.similarity_fn_descr == 'neuron':
            self.similarity_fn = _per_neuron_similarity_fn
        elif self.similarity_fn_descr == 'activity':
            self.similarity_fn = _activity_similarity_fn
        else:
            raise ValueError(f'Did not recognize loss_function! Given: {self.similarity_fn_descr}')

    @staticmethod
    def create_from_config(config, model):
        return ParallelSearch(
            model=model,
            num_bins=config['num_bins'],
            T_l = config['encoding_length'],
            input_dim = config['input_dim'],
            num_classes = config['output_dim'],
            similarity_fn_descr = config['similarity_function'],
            parallel_beams = config['parallel_beams', 5],
        )

    def forward(self, x, lens):
        B = x.shape[0]
        T = x.shape[1]
        J = self.input_dim
        C = self.num_classes
        N = self.parallel_beams
        T_l = self.T_l
        device = x.device
        
        label_seqs = self.model.spikoder.get_all_labels()

        sos = self.model.special_token_encoder()
        sos = sos.repeat(B, 1)

        x = self.place_preceeding_and_following_sos(x, lens, sos)
        T = T + 1 # Because we placed sos at the beginning

        # Those are the timesteps at which we can 
        # find our prediction per batch entry. 
        idx = torch.arange(T_l).view(1, T_l).to(device)
        idx = idx + lens.unsqueeze(1).repeat(1, T_l) + 1
        idx = idx.unsqueeze(-1).repeat(1,1,J).long()

        assert list(idx.shape) == [B, T_l, J]

        # We do the first step here to initialize pred_label_seq.
        # pred_label_seq will contain the predicted label sequence.
        pred = self.model.search_forward(x)
        pred_label_seq = torch.gather(
            pred,
            1, 
            idx[:,0,:].unsqueeze(1)
        )

        n_best = self.get_n_most_similar_label_sofar(
            pred_label_seq, label_seqs, 1
        )

        assert list(n_best.shape) == [B, N]

        x, idx = self.fill_with_n_best_from_first_step(x, idx, n_best, label_seqs)

        assert list(x.shape) == [B*N, T, J]
        assert list(idx.shape) == [B*N, T_l, J]

        pred = self.model.search_forward(x)
        pred_label_seq = torch.gather(
            pred,
            1, 
            idx
        )

        assert list(pred_label_seq.shape) == [B*N, T_l, J]

        pred = self.extract_most_similar_label(pred_label_seq, label_seqs, B)

        return pred, None


    def place_preceeding_and_following_sos(self, x, lens, sos):
        # Todo: Ideally this is done before batching
        for b in range(x.shape[0]):
            x[b,lens[b],:] = sos[b]
        sos = sos.unsqueeze(1)
        x = torch.cat([sos, x], dim=1)
        return x

    def get_n_most_similar_label_sofar(self, pred_label_seq, label_seqs, t):
        B = pred_label_seq.shape[0]
        J = self.input_dim
        C = self.num_classes

        label_seqs = label_seqs[:,:t,:]
        
        assert list(label_seqs.shape) == [C, t, J]
        assert list(pred_label_seq.shape) == [B, t, J]

        label_seqs = label_seqs.unsqueeze(0)
        pred_label_seq = pred_label_seq.unsqueeze(1)
        similarity = self.similarity_fn(pred_label_seq, label_seqs)

        assert list(similarity.shape) == [B, C, t]

        similarity = torch.mean(similarity, dim=-1)
        predicted_label_sofar = torch.topk(similarity, self.parallel_beams, dim=-1, largest=False)

        return predicted_label_sofar[1]

    def fill_with_n_best_from_first_step(self, x, idx, n_best, label_seqs):
        B = x.shape[0]
        T = x.shape[1]
        J = self.input_dim
        C = self.num_classes
        N = self.parallel_beams
        T_l = self.T_l

        x = torch.repeat_interleave(x, N, dim=0)
        idx = torch.repeat_interleave(idx, N, dim=0)
        n_best = n_best.view(B*N)

        for b in range(B*N):
            for t in range(T_l):
                x[b, idx[b,t]] = label_seqs[n_best[b], t]

        return x, idx
    
    def extract_most_similar_label(self, pred_label_seq, label_seqs, B):
        J = self.input_dim
        C = self.num_classes
        N = self.parallel_beams
        T_l = self.T_l
        BN = B * N
        

        assert list(label_seqs.shape) == [C, T_l, J]
        assert list(pred_label_seq.shape) == [BN, T_l, J]

        label_seqs = label_seqs.unsqueeze(0)
        pred_label_seq = pred_label_seq.unsqueeze(1)
        similarity = self.similarity_fn(pred_label_seq, label_seqs)

        assert list(similarity.shape) == [BN, C, T_l]

        similarity = torch.mean(similarity, dim=-1)
        similarity = similarity.view(B, N, C)

        valuesC, indicesC = torch.min(similarity, dim=-1)
        _, indicesN = torch.min(valuesC, dim=-1)

        label = torch.gather(indicesC, -1, indicesN.unsqueeze(-1))

        return label.squeeze()