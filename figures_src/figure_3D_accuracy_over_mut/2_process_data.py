"""
Figure 3D — Step 2: fit decay rates across mutation rates.

Loads the mutation-rate accuracy sweep, normalises each trajectory by its
first time-point, and fits an exponential decay rate rho via
get_single_decay_rate(..., mut=muts[i]) for each (mut_rate, replicate) pair.
Each row uses its own mutation rate when fitting so the exponential model is
correctly parameterised.

Input
-----
  SLIDE_data/mutation_rate_accuracy.pkl
    Raw decay curves.  Reshaped to (25 mut rates, n_curves, 25 steps);
    n_curves is typically 500 (50 reps × 10 instances).

Output
------
  plot_data/mut_accuracy.pkl
    Tuple (mut_decay_rates, muts):
      mut_decay_rates : ndarray (25, n_curves) — fitted rho per replicate
      muts            : ndarray (25,)           — mutation rates,
                        linspace(0.01, 2, 25)

Source
------
  ruggedness_figures_data_processing.ipynb, cells 21–24.
"""

import os
import sys
import pickle
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

from ruggedness_functions import get_single_decay_rate
from slide_config import get_slide_data_dir

slide_data_dir = get_slide_data_dir()
plot_data_dir  = os.path.join(parent_dir, 'plot_data')
os.makedirs(plot_data_dir, exist_ok=True)

out = os.path.join(plot_data_dir, 'mut_accuracy.pkl')

if os.path.exists(out):
    print(f'Already exists: {out}')
    sys.exit(0)

# --- Load raw decay curves ---

with open(os.path.join(slide_data_dir, 'mutation_rate_accuracy.pkl'), 'rb') as f:
    mut_data = pickle.load(f)

# Reshape to (25 mut rates, n_curves, 25 steps) and normalise by first step
reshaped = mut_data.reshape(25, -1, 25)
curves   = reshaped / reshaped[:, :, 0][:, :, np.newaxis]

# Mutation rates — must match what was used when generating the sweep
muts = np.linspace(0.01, 2, 25)

# --- Fit decay rates ---

n_muts   = curves.shape[0]
n_curves = curves.shape[1]

mut_decay_rates = np.zeros((n_muts, n_curves))
for i in range(n_muts):
    for ii in range(n_curves):
        mut_decay_rates[i, ii] = get_single_decay_rate(curves[i, ii, :], mut=muts[i])[0]

# --- Save ---

with open(out, 'wb') as f:
    pickle.dump((mut_decay_rates, muts), f)

print(f'Saved → {out}')
