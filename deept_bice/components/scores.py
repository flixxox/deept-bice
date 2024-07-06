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