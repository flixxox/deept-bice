import torch
from colour import Color

import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec

from deept.utils.setup import setup
from deept.train import create_model
from deept.utils.config import Config
from deept.train import create_dataloader
from deept.utils.globals import Settings, Context
from deept.utils.checkpoint_manager import CheckpointManager

from deept_bice.components.seeker import GreedySearch


# === Setup

config = Config.parse_config({
    'config': '/Scratch/fschmidt/output/bice/shd/delayLmLIF/workflow-v3-population-lambdaData0-lambdaLabel1-150724_183200/input/config.yaml'
})

config['user_code'] = '/Users/fschmidt/code/deept-bice'
config['batch_size'] = 256
config['number_of_gpus'] = 1
config['load_weights'] = True

config['load_weights_from'] = '/Scratch/fschmidt/output/bice/shd/delayLmLIF/workflow-v3-population-lambdaData0-lambdaLabel1-150724_183200/work/train_jobs/TrainJob.O41nuFFDFyPH/output/seed_0000/checkpoints/ckpt-47.pt'

config.print_config()

setup(config, 0, 1, 
    train=False,
    time=False
)

model = create_model(config)
Context.add_context('model', model)

checkpoint_manager = checkpoint_manager = CheckpointManager.create_eval_checkpoint_manager_from_config(config)
checkpoint_manager.restore_if_requested()
model = Context['model']
model = model.to(Settings.get_device())
model.eval()
print(model)

train_dataloader, dev_dataloader = create_dataloader(config)

search_algorithm = GreedySearch.create_from_config(config, model)

search_algorithm.to(Settings.get_device())

# === Search

T_l = config['encoding_length']

numel_accum = 0
accs_accum = [0. for t in range(T_l)]
redecisions_accum = [0. for t in range(T_l)]

with torch.no_grad():
    for data in dev_dataloader:

        data = data['tensors']

        x = data['inp'].to(Settings.get_device())
        lens = data['lens'].to(Settings.get_device())
        labels = data['labels'].to(Settings.get_device())
        B = x.shape[0]

        pred_label_prev = None
        for t in range(T_l):
            pred_label_cur, _ = search_algorithm(x, lens, early_abort_at=t+1)
            acc = (pred_label_cur == labels).sum() * 100
            accs_accum[t] += acc

            if pred_label_prev is not None:
                redecisions = (pred_label_cur != pred_label_prev).sum()
                redecisions_accum[t] += redecisions

            pred_label_prev = pred_label_cur

        numel_accum += B

# === Print

for t in range(T_l):
    avg_test_acc_t = accs_accum[t] / numel_accum
    avg_redecisions_t = redecisions_accum[t] / numel_accum
    print(f'Avg Test Acc {t}: {avg_test_acc_t:4.2f}, Percentage of Redecisions {avg_redecisions_t*100:4.2f}%!')

print('Avg test_acc')
for t in range(T_l):
    avg_test_acc_t = accs_accum[t] / numel_accum
    print(f'({t+1},{avg_test_acc_t:4.2f})', end='')
print('\nAvg redecisions')
for t in range(T_l):
    avg_redecisions_t = redecisions_accum[t] / numel_accum
    print(f'({t+1},{avg_redecisions_t*100:4.2f})', end='')
print('\n')


