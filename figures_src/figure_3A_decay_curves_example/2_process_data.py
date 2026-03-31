"""
Figure 3A — Step 2: run directed evolution on smooth/rugged NK landscapes and
fit exponential decay rates.

Runs directed evolution on two NK pairs:
  - (N=20, K=14)  rugged landscape
  - (N=20, K=1)   smooth landscape

Normalises the mean fitness trajectories to decay from 1, fits exponential
decay rates via get_single_decay_rate(), then computes the fitted line arrays
for plotting.

Originally from ruggedness_figures_data_processing.ipynb cells 8–9.

Input
-----
  None — NK landscapes are generated on-the-fly from a fixed PRNG key.

Output
------
  plot_data/smooth_rugged_example.pkl
    (smooth_rugged, fitted_lines)
      smooth_rugged : np.ndarray of shape (2, 25) — normalised mean fitness
                      per generation (rugged first, smooth second)
      fitted_lines  : np.ndarray of shape (2, 25) — fitted exponential decay
                      curves
"""

import os
import sys
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

from direvo_functions import *
from ruggedness_functions import get_single_decay_rate
import selection_function_library as slct

plot_data_dir = os.path.join(parent_dir, 'plot_data')
os.makedirs(plot_data_dir, exist_ok=True)

out = os.path.join(plot_data_dir, 'smooth_rugged_example.pkl')

if os.path.exists(out):
    print(f'Already exists: {out}')
    sys.exit(0)

NK_pair = [(20, 14), (20, 1)]
out_curves = []
params = {'threshold': 0.0, 'base_chance': 1.0}

for nk in NK_pair:
    run = directedEvolution(
        jr.PRNGKey(0),
        N=int(nk[0]), K=int(nk[1]),
        selection_strategy=slct.base_chance_threshold_select,
        selection_params=params,
        popsize=300,
        mut_chance=0.5 / int(nk[0]),
        num_steps=25, num_reps=1,
        pre_optimisation_steps=20,
        average=True,
    )
    out_curves.append(run['fitness'].mean(axis=-1))

smooth_rugged = np.array([out_curves[i][0] / out_curves[i][0][0] for i in range(2)])
smooth_rugged_decay_rates = np.array([get_single_decay_rate(i, mut=0.5) for i in smooth_rugged])

mutations = np.linspace(0.0, 12, 25)
fitted_lines = np.array([
    np.exp(-mutations * i[0]) * (1 - i[-1]) + i[-1]
    for i in smooth_rugged_decay_rates
])

with open(out, 'wb') as f:
    pickle.dump((smooth_rugged, fitted_lines), f)

print(f'Saved → {out}')
