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

from deept_bice.components.scores import (
    _activity_similarity_fn,
    _per_neuron_similarity_fn
)


# === Setup

config = Config.parse_config({
    'config': '/Scratch/fschmidt/output/bice/shd/delayLmLIF/workflow-v0-population-originalOptim-lambda_data_1-lambda_label_1-hiddenSize_1024-nLayer_3-130724_101324/input/config.yaml'
})

config['batch_size'] = 1
config['number_of_gpus'] = 1
config['load_weights'] = True
config['user_code'] = '/Users/fschmidt/code/deept-bice'

config['load_weights_from'] = '/Scratch/fschmidt/output/bice/shd/delayLmLIF/workflow-v0-population-originalOptim-lambda_data_1-lambda_label_1-hiddenSize_1024-nLayer_3-130724_101324/work/train_jobs/TrainJob.7EH2oc7z6fSD/output/seed_0000/checkpoints/ckpt-146.pt'

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
model = model.eval()
print(model)

train_dataloader, dev_dataloader = create_dataloader(config)

# === Get activities

C = config['output_dim']
activities_tgt = [[] for c in range(C)]
activities_out = [[] for c in range(C)]
similarities = [[] for c in range(C)]

count = 0
with torch.no_grad():
    for data in train_dataloader:
        
        if isinstance(data, list):
            data = data[0]

        data = data['tensors']
        
        label = data['labels'].cpu().item()

        if len(activities_tgt[label]) == 0:
            for key, tensor in data.items():
                data[key] = tensor.to(Settings.get_device())

            out, add_out = model(data['inp'], data['tgt'], data['lens'], data['labels'])
            tgt = torch.clamp(add_out['tgt'], max=1)

            activities_tgt[label] = list(torch.mean(tgt, dim=-1).squeeze().detach().cpu().numpy())
            activities_out[label] = list(torch.mean(out, dim=-1).squeeze().detach().cpu().numpy())
            
            similarities[label] = list(_per_neuron_similarity_fn(out, tgt).squeeze().detach().cpu().numpy())
            
            count += 1

        if count >= C:
            break

# === Print

for c in range(C):
    avg_sim = sum(similarities[c])/len(similarities[c])
    print(f'Label {c}: avg_sim {avg_sim*100}!')

# === Plot

plt.title('Activity of Label Encodings')

gs = (grid_spec.GridSpec(C,1))
fig = plt.figure(figsize=(8,40))

for c in range(C):

    ax = fig.add_subplot(gs[c:c+1, 0:])

    ys_tgt = activities_tgt[c]
    ys_out = activities_out[c]
    ys_sim = similarities[c]
    xs = [i for i in range(1,len(ys_tgt)+1)]

    ax.plot(xs, ys_tgt, color='green', label=f'tgt {c+1}')
    ax.plot(xs, ys_out, color='red', label=f'out {c+1}')
    ax.plot(xs, ys_sim, color='blue', label=f'out {c+1}')
    ax.legend()
    
plt.tight_layout()
plt.savefig('activity_model_tgt.png', bbox_inches='tight', dpi=200)