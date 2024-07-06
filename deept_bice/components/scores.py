import torch

from deept.components.scores import (
    Score,
    register_score
)

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
            self.loss_fn = self._per_neuron_loss
        elif self.loss_function_descr == 'activity':
            self.loss_fn = self._activity_loss
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

    def __call__(self, output, data, label, sos):
        B = data.shape[0]
        J = self.input_dim
        T_d = data.shape[1]
        T_l = self.encoding_length

        assert list(sos.shape) == [B, J]
        assert list(data.shape) == [B, T_d, J]
        assert list(output.shape) == [B, T_d+T_l+2, J]
        assert list(label.shape) == [B, T_l, J]

        loss = self.loss_fn(output, data, label)
        
        self.accumulators[0].increase(loss, numel)

        return None, None

    def _per_neuron_loss(self, output, data, label):
        return 0.
    
    def _activity_loss(self, output, data, label):
        return 0.
