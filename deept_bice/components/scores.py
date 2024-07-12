import torch
import torch.nn as nn

from deept.utils.globals import Context
from deept.components.scores import (
    Score,
    register_score
)


def _per_neuron_similarity_fn(out, tgt, reduce=True):
        loss = torch.abs(out - tgt)
        # Always reduce feature dim.
        loss = torch.mean(loss, dim=-1)
        return loss
    
def _activity_similarity_fn(out, tgt, reduce=True):
    out = torch.mean(out, dim=-1)
    tgt = torch.mean(tgt, dim=-1)
    loss = torch.pow(out - tgt, 2)
    return loss


@register_score('LMRegressionLoss')
class LMRegressionLoss(Score):

    def __init__(self, input_keys, reduce_type, **kwargs):
        super().__init__(input_keys, reduce_type)

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.register_accumulator('lmloss')

        self.input_dim = self.input_dim // self.num_bins

        if self.similarity_function_descr == 'neuron':
            self.similarity_fn = _per_neuron_similarity_fn
        elif self.similarity_function_descr == 'activity':
            self.similarity_fn = _activity_similarity_fn
        else:
            raise ValueError(f'Did not recognize loss_function! Given: {self.similarity_function_descr}')

    @staticmethod
    def create_from_config(config, input_keys, reduce_type):
        return LMRegressionLoss(
            input_keys, reduce_type,
            num_bins = config['num_bins'],
            input_dim = config['input_dim'],
            encoding_length = config['encoding_length'],
            similarity_function_descr = config['similarity_function'],
            clamp_tgt_to_one = config['clamp_tgt_to_one'],
            lambda_data = config['lambda_data'],
            lambda_label = config['lambda_label'],
        )

    def __call__(self, out, tgt, mask, label_mask):
        B = out.shape[0]
        T = out.shape[1]
        J = self.input_dim
        lambda_data = self.lambda_data
        lambda_label = self.lambda_label

        assert list(tgt.shape) == [B, T, J]
        assert list(out.shape) == [B, T, J]
        assert list(mask.shape) == [B, T]
        assert list(label_mask.shape) == [B, T]

        if self.clamp_tgt_to_one:
            tgt = torch.clamp(tgt, max=1)

        loss = self.similarity_fn(out, tgt)

        assert list(loss.shape) == [B, T]
        
        data_mask = mask - label_mask
        numel = (lambda_data * data_mask.sum()
            + lambda_label * label_mask.sum())

        loss = loss * mask # Remove padding
        loss = (lambda_data * data_mask * loss
            + lambda_label * label_mask * loss)
        loss = loss.sum() / numel
        
        # The loss is super small, we
        # might get underflowing gradients
        loss *= 100

        self.accumulators[0].increase(loss, numel)

        return loss, numel


@register_score('LMAccuracy')
class LMAccuracy(Score):

    def __init__(self, input_keys, reduce_type, **kwargs):
        super().__init__(input_keys, reduce_type)

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.input_dim = self.input_dim // self.num_bins
        
        if self.similarity_fn_descr == 'neuron':
            self.similarity_fn = _per_neuron_similarity_fn
        elif self.similarity_fn_descr == 'activity':
            self.similarity_fn = _activity_similarity_fn
        else:
            raise ValueError(f'Did not recognize loss_function! Given: {self.similarity_fn_descr}')

        self.idx = nn.Parameter(
            torch.arange(self.encoding_length),
            requires_grad=False
        )
        
        self.register_accumulator('lm_acc')

    @staticmethod
    def create_from_config(config, input_keys, reduce_type):
        return LMAccuracy(
            input_keys, reduce_type,
            num_bins = config['num_bins'],
            input_dim = config['input_dim'],
            num_classes = config['output_dim'],
            encoding_length = config['encoding_length'],
            similarity_fn_descr = config['similarity_function']
        )

    def __call__(self, out, labels, label_seqs, lens):
        B = out.shape[0]
        T = out.shape[1]
        J = self.input_dim
        C = self.num_classes
        T_l = self.encoding_length
        
        assert list(labels.shape) == [B]
        assert list(label_seqs.shape) == [C, T_l, J]

        # Here we extract \hat{l} from the output sequence.
        # We need to take padding and the last sos into account.
        # We base the operation on torch.gather, which
        # works like a index_select over multiple dims.

        idx = self.idx.unsqueeze(0).repeat(B, 1)
        idx = idx + lens.unsqueeze(1) + 1
        idx = idx.unsqueeze(-1).repeat(1, 1, J)

        out = torch.gather(out, 1, idx)

        assert list(out.shape) == [B, T_l, J]

        out = out.unsqueeze(1)
        label_seqs = label_seqs.unsqueeze(0)

        loss = self.similarity_fn(out, label_seqs, reduce=False)

        assert list(loss.shape) == [B, C, T_l]

        loss = torch.mean(loss, dim=-1)

        numel = B
        pred = torch.argmin(loss, dim=-1)

        assert list(pred.shape) == [B]

        acc = (pred == labels).sum() * 100
        acc = acc / numel

        self.accumulators[0].increase(acc, numel)
        
        return acc, numel


@register_score('LMTrueAccuracy')
class LMTrueAccuracy(Score):

    def __init__(self, input_keys, reduce_type, **kwargs):
        super().__init__(input_keys, reduce_type)

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.input_dim = self.input_dim // self.num_bins
        
        if self.similarity_fn_descr == 'neuron':
            self.similarity_fn = _per_neuron_similarity_fn
        elif self.similarity_fn_descr == 'activity':
            self.similarity_fn = _activity_similarity_fn
        else:
            raise ValueError(f'Did not recognize loss_function! Given: {self.similarity_fn_descr}')

        self.idx = nn.Parameter(
            torch.arange(self.encoding_length),
            requires_grad=False
        )
        
        self.register_accumulator('search_acc')

    @staticmethod
    def create_from_config(config, input_keys, reduce_type):

        search_algorithm_descr = config['lm_search_algorithm']

        if search_algorithm_descr == 'greedy':
            from deept_bice.components.seeker import GreedySearch
            search_algorithm = GreedySearch.create_from_config(config, Context['model'])
        else:
            raise ValueError(f'Unrecognized search algorithm {search_algorithm_descr}!')

        return LMTrueAccuracy(
            input_keys, reduce_type,
            search_algorithm = search_algorithm,
            num_bins = config['num_bins'],
            input_dim = config['input_dim'],
            num_classes = config['output_dim'],
            encoding_length = config['encoding_length'],
            similarity_fn_descr = config['similarity_function'],
        )

    def __call__(self, out, x, lens, labels):
        if not self.search_algorithm.model.training:
            B = out.shape[0]
            T = out.shape[1]
            J = self.input_dim
            C = self.num_classes
            T_l = self.encoding_length
            
            pred_label, _ = self.search_algorithm(x, lens)

            numel = B
            acc = (pred_label == labels).sum() * 100
            acc = acc / numel

            self.accumulators[0].increase(acc, numel)
            
            return acc, numel
        else:
            self.accumulators[0].increase(0, 1)
            return 0, 1