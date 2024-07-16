import torch
import numpy as np
import seaborn as sns

from scipy.stats import shapiro
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec

from deept.utils.setup import setup
from deept.train import create_model
from deept.utils.config import Config
from deept.utils.globals import Settings, Context
from deept.utils.checkpoint_manager import CheckpointManager


# === Setup

config = Config.parse_config({
    # lambdaData = 1, lambdaLabel = 1 
    #'config': '/Scratch/fschmidt/output/bice/shd/delayLmLIF/workflow-v3-population-lambdaData1-lambdaLabel1-150724_162946/input/config.yaml'
    # lambdaData = 0, lambdaLabel = 1 
    'config': '/Scratch/fschmidt/output/bice/shd/delayLmLIF/workflow-v3-population-lambdaData0-lambdaLabel1-150724_183200/input/config.yaml'
    # lambdaData = 1, lambdaLabel = 0
    #'config': '/Scratch/fschmidt/output/bice/shd/delayLmLIF/workflow-v3-population-lambdaData1-lambdaLabel0-150724_183220/input/config.yaml'
})

config['batch_size'] = 256
config.print_config()
config['user_code'] = '/Users/fschmidt/code/deept-bice'
config['number_of_gpus'] = 0
config['load_weights'] = True

# lambdaData = 1, lambdaLabel = 1 
#config['load_weights_from'] = '/Scratch/fschmidt/output/bice/shd/delayLmLIF/workflow-v3-population-lambdaData1-lambdaLabel1-150724_162946/work/train_jobs/TrainJob.EhmxErJyfHV4/output/seed_0000/checkpoints/ckpt-20.pt'
# lambdaData = 0, lambdaLabel = 1 
config['load_weights_from'] = '/Scratch/fschmidt/output/bice/shd/delayLmLIF/workflow-v3-population-lambdaData0-lambdaLabel1-150724_183200/work/train_jobs/TrainJob.O41nuFFDFyPH/output/seed_0000/checkpoints/ckpt-46.pt'
# lambdaData = 1, lambdaLabel = 0 
#config['load_weights_from'] = '/Scratch/fschmidt/output/bice/shd/delayLmLIF/workflow-v3-population-lambdaData1-lambdaLabel0-150724_183220/work/train_jobs/TrainJob.TkeiF2zsJfmC/output/seed_0000/checkpoints/ckpt-55.pt'

setup(config, 0, 1, 
    train=False,
    time=False
)

model = create_model(config)
Context.add_context('model', model)

checkpoint_manager = checkpoint_manager = CheckpointManager.create_eval_checkpoint_manager_from_config(config)
checkpoint_manager.restore_if_requested()
model = Context['model']
model.eval()
print(model)

for lif_node in model.lif_nodes:
    lif_node.delay.SIG *= 0
    lif_node.delay.version = 'max'
    lif_node.delay.DCK.version = 'max'
model.round_delay()

D_max = config['max_delay']
delays = [] 
for k, v in model.state_dict().items():
    if 'delay.P' in k:
        v = v
        v = list(v.squeeze().flatten().numpy())
        delays += v

delays = np.array(delays)

# === Plot

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)

plot = sns.kdeplot(delays, ax=ax)

plt.tight_layout()
plt.savefig('test.png', bbox_inches='tight')
