import math

import torch
from colour import Color

import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec

from deept_bice.components.spikoder import create_spikoder

T = 10
C = 20
J = 140

spikoder = create_spikoder(
    'sinusoidal',
    J,
    C,
    T,
    False, # sample
    False, # sample_every_step
)

labels = spikoder.get_all_labels()

assert list(labels.shape) == [C, T, J]

labels = torch.mean(labels, dim=-1)
labels = list(labels.numpy())

# ====  Print

red = Color("#C1E899")
colors = list(red.range_to(Color("#55883B"), C))

for c in range(C//2):

    xs = [i for i in range(1,T+1)]
    ys = labels[c]
    
    print(
        f'\\addplot '
        'coordinates { '
    )
    for x,y in zip(xs, ys):
        x = (math.ceil(x*100)/100)
        print(f'({x},{y})', end='')
    print('};')


# ==== Plot

plt.title('Activity of Label Encodings')

gs = (grid_spec.GridSpec(C,1))
fig = plt.figure(figsize=(8,40))

for c in range(C):

    ax = fig.add_subplot(gs[c:c+1, 0:])

    xs = [i for i in range(1,T+1)]
    ys = labels[c]

    ax.plot(xs, ys, color=colors[c].hex, label=f'label {c+1}')
    ax.legend()
    
plt.tight_layout()
plt.savefig('test.png', bbox_inches='tight', dpi=200)