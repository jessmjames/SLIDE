"""
Figure 3B — Step 2: fit decay rates across the NK ruggedness grid.

Loads the large decay-curve sweep over 100 (N, K) pairs, normalises each
trajectory by its first time-point, fits an exponential decay rate rho via
get_single_decay_rate(), and computes the (K+1)/N ruggedness label used on
the x-axis of the figure.

Input
-----
  SLIDE_data/large_decay_curve_sweep.pkl
    Raw decay curves.  Shape: (100, reps*instances, 25) after reshape, where
    the leading 100 dimension indexes NK pairs from NK_grid([10, 50]).

Output
------
  plot_data/ruggedness_accuracy.pkl
    Tuple (k_plus_one_over_ns, decay_rates):
      k_plus_one_over_ns : ndarray (100,)        — (K+1)/N for each NK pair
      decay_rates        : ndarray (100, n_curves) — fitted rho per replicate

Source
------
  ruggedness_figures_data_processing.ipynb, cells 5–14.
"""

import os
import sys
import pickle
import numpy as np

import jax.numpy as jnp

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

from ruggedness_functions import get_single_decay_rate
from slide_config import get_slide_data_dir

slide_data_dir = get_slide_data_dir()
plot_data_dir  = os.path.join(parent_dir, 'plot_data')
os.makedirs(plot_data_dir, exist_ok=True)

out = os.path.join(plot_data_dir, 'ruggedness_accuracy.pkl')

if os.path.exists(out):
    print(f'Already exists: {out}')
    sys.exit(0)

# --- Build the NK grid (same as notebook cells 5–9) ---

def NK_grid(N_range, num_samples=10):
    N = jnp.linspace(N_range[0], N_range[1], num=num_samples)
    K = jnp.array([jnp.linspace(1, i, num_samples) for i in N]).reshape(num_samples, num_samples)
    N = jnp.repeat(N, num_samples).reshape(num_samples, num_samples)
    return N, K

N_grid, K_grid = NK_grid([10, 50])
Ns = jnp.flip(N_grid.flatten())
Ks = jnp.flip(K_grid.flatten())
NKs = list(zip(Ns, Ks))

# --- Load raw decay curves ---

with open(os.path.join(slide_data_dir, 'large_decay_curve_sweep.pkl'), 'rb') as f:
    decay_data = pickle.load(f)

# Reshape to (100 NK pairs, n_curves, 25 steps) and normalise by first step
reshaped = decay_data.reshape(100, -1, 25)
normalized = reshaped / reshaped[:, :, 0][:, :, np.newaxis]

# --- Fit decay rates ---

n_nk     = normalized.shape[0]
n_curves = normalized.shape[1]

decay_rates = np.zeros((n_nk, n_curves))
for i in range(n_nk):
    for ii in range(n_curves):
        decay_rates[i, ii] = get_single_decay_rate(normalized[i, ii, :], mut=0.5)[0]

# (K+1)/N ruggedness label, clipped to [0, 1]
k_plus_one_over_ns = np.clip(
    (np.array(NKs)[:, 1] + 1) / np.array(NKs)[:, 0], 0, 1
)

# --- Save ---

with open(out, 'wb') as f:
    pickle.dump((k_plus_one_over_ns, decay_rates), f)

print(f'Saved → {out}')
