import torch

from deept.components.scores import (
    Score,
    register_score
)


def _per_neuron_similarity_fn(out, tgt, reduce=True):
        numel = out.numel()

        loss = torch.abs(out - tgt)
        # Always reduce feature dim
        loss = torch.mean(loss, dim=-1)
        
        if reduce:
            loss = torch.mean(loss)

        return loss, numel
    
def _activity_similarity_fn(out, tgt, reduce=True):
    numel = out.numel()

    out = torch.mean(out, dim=-1)
    tgt = torch.mean(tgt, dim=-1)
    loss = torch.sqrt(out - tgt)

    if reduce:
        loss = torch.mean(loss)

    return loss, numel


@register_score('Accuracy')
class Accuracy(Score):

    def __init__(self, input_keys, reduce_type):
        super().__init__(input_keys, reduce_type)
        self.register_accumulator('acc')

    @staticmethod
    def create_from_config(config, input_keys, reduce_type):
        return Accuracy(
            input_keys, reduce_type
        )

    def __call__(self, output, targets):
        _, idx = output.max(1)
        numel = targets.numel()
        acc = (targets == idx).sum() * 100
        acc = acc / numel
        self.accumulators[0].increase(acc, numel)
        return acc, numel


@register_score('FiringRate')
class FiringRate(Score):

    def __init__(self, input_keys, reduce_type,
        time_step_duration=10
    ):
        super().__init__(input_keys, reduce_type)
        self.time_step_duration = time_step_duration
        self.register_accumulator('fr')

    @staticmethod
    def create_from_config(config, input_keys, reduce_type):
        return FiringRate(
            input_keys, reduce_type,
            time_step_duration = config['time_step']
        )

    def __call__(self, output, all_spikes):
        numel = len(all_spikes)
        fr_accum = 0.
        for x in all_spikes:
            time_steps = x.shape[1]
            fr = x.sum(dim=1) / (time_steps * self.time_step_duration)
            fr = fr.mean()
            fr_accum += fr
        fr_accum = (fr_accum / numel) * 1000
        self.accumulators[0].increase(fr_accum, numel)
        return None, None


@register_score('LMRegressionLoss')
class LMRegressionLoss(Score):

    def __init__(self, input_keys, reduce_type, **kwargs):
        super().__init__(input_keys, reduce_type)

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.register_accumulator('lmloss')

        self.input_dim = self.input_dim // self.num_bins

        if self.loss_function_descr == 'neuron':
            self.loss_fn = _per_neuron_similarity_fn
        elif self.loss_function_descr == 'activity':
            self.loss_fn = _activity_similarity_fn
        else:
            raise ValueError(f'Did not recognize loss_function! Given: {self.loss_function_descr}')

    @staticmethod
    def create_from_config(config, input_keys, reduce_type):
        return LMRegressionLoss(
            input_keys, reduce_type,
            num_bins = config['num_bins'],
            input_dim = config['input_dim'],
            encoding_length = config['encoding_length'],
            loss_function_descr = config['criterions'][0]['loss_function']
        )

    def __call__(self, out, tgt):
        B = out.shape[0]
        T = out.shape[1]
        J = self.input_dim

        assert list(tgt.shape) == [B, T, J]
        assert list(out.shape) == [B, T, J]

        loss, numel = self.loss_fn(out, tgt)
        
        self.accumulators[0].increase(loss, numel)

        loss = loss / numel

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
        
        self.register_accumulator('lm_acc')

    @staticmethod
    def create_from_config(config, input_keys, reduce_type):
        return LMAccuracy(
            input_keys, reduce_type,
            num_bins = config['num_bins'],
            input_dim = config['input_dim'],
            num_classes = config['output_dim'],
            encoding_length = config['encoding_length'],
            similarity_fn_descr = config['criterions'][0]['loss_function']
        )

    def __call__(self, out, labels, label_seqs):
        B = out.shape[0]
        T = out.shape[1]
        J = self.input_dim
        C = self.num_classes
        T_l = self.encoding_length
        
        assert list(labels.shape) == [B]
        assert list(label_seqs.shape) == [C, T_l, J]
        
        # Extract \hat{l} without the last symbol
        # because it is predicting the sos token
        out = out[:,T-T_l-1:-1,:]

        assert list(out.shape) == [B, T_l, J]

        out = out.unsqueeze(1)
        label_seqs = label_seqs.unsqueeze(0)

        loss, _ = self.similarity_fn(out, label_seqs, reduce=False)

        assert list(loss.shape) == [B, C, T_l]

        loss = torch.mean(loss, dim=-1)
        
        numel = B
        pred = torch.argmax(loss, dim=-1)

        assert list(pred.shape) == [B]

        acc = (pred == labels).sum() * 100
        acc = acc / numel

        self.accumulators[0].increase(acc, numel)
        
        return acc, numel