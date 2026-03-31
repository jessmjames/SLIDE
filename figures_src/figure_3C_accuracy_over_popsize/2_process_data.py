"""
Figure 3C — Step 2: fit decay rates across population sizes.

Loads the population-size accuracy sweep, normalises each trajectory by its
first time-point, and fits an exponential decay rate rho via
get_single_decay_rate() for each (popsize, replicate) pair.

Input
-----
  SLIDE_data/popsize_accuracy.pkl
    Raw decay curves.  Reshaped to (25 pop sizes, n_curves, 25 steps);
    n_curves is typically 500 (50 reps × 10 instances).

Output
------
  plot_data/popsize_accuracy.pkl
    Tuple (popsize_decay_rates, pops):
      popsize_decay_rates : ndarray (25, n_curves) — fitted rho per replicate
      pops                : ndarray (25,)           — population sizes,
                            linspace(100, 2500, 25, dtype=int)

Source
------
  ruggedness_figures_data_processing.ipynb, cells 16–19.
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

out = os.path.join(plot_data_dir, 'popsize_accuracy.pkl')

if os.path.exists(out):
    print(f'Already exists: {out}')
    sys.exit(0)

# --- Load raw decay curves ---

with open(os.path.join(slide_data_dir, 'popsize_accuracy.pkl'), 'rb') as f:
    popsize_data = pickle.load(f)

# Reshape to (25 pop sizes, n_curves, 25 steps) and normalise by first step
reshaped = popsize_data.reshape(25, -1, 25)
curves   = reshaped / reshaped[:, :, 0][:, :, np.newaxis]

# --- Fit decay rates ---

n_pops   = curves.shape[0]
n_curves = curves.shape[1]

popsize_decay_rates = np.zeros((n_pops, n_curves))
for i in range(n_pops):
    for ii in range(n_curves):
        popsize_decay_rates[i, ii] = get_single_decay_rate(curves[i, ii, :], mut=1.0)[0]

pops = np.linspace(100, 2500, 25, dtype=int)

# --- Save ---

with open(out, 'wb') as f:
    pickle.dump((popsize_decay_rates, pops), f)

print(f'Saved → {out}')
