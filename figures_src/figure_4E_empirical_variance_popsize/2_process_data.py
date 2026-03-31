"""
Figure 4E — Step 2: compute estimation_variance.pkl (shared with Fig 4F).

Loads per-landscape popsize decay curves and computes bootstrap rho estimates
over population sizes (array_results1) and over number of generations sampled
(array_results2), then saves both together.

Input
-----
  SLIDE_data/decay_curves_GB1_m0.1_multistart_10000_uniform_multi_popsize.pkl
  SLIDE_data/decay_curves_TrpB_m0.1_multistart_10000_uniform_multi_popsize.pkl
  SLIDE_data/decay_curves_TEV_m0.1_multistart_10000_uniform_multi_popsize.pkl
  SLIDE_data/decay_curves_ParD3_m0.1_multistart_10000_uniform_multi_popsize.pkl
    Each file is reshaped to (10 popsizes, n_starts, 10 replicates, 25 generations).

Output
------
  plot_data/estimation_variance.pkl
    (array_results1, array_results2)
    array_results1: list of 4 arrays, shape (7, n_sub) — bootstrap rho per
      popsize (popsizes index 3:10) for GB1, TrpB, TEV, ParD3.
    array_results2: list of 4 arrays, shape (23, n_boot) — bootstrap rho per
      number of generations (2:25) at the highest popsize for each landscape.

Note
----
  Originally from ruggedness_figures_data_processing.ipynb.
  Load cell: popsizes = np.linspace(25, 2500, 10) + 4 file loads.
  array_results1 cell: n_sub = 250, np.random.seed(42) — bootstrap over popsizes.
  array_results2 cell: n_boot = 250, np.random.seed(42) — bootstrap over generations.
  Save cell: pickle.dump((array_results1, array_results2), f).
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

out = os.path.join(plot_data_dir, 'estimation_variance.pkl')

if os.path.exists(out):
    print(f'Already exists: {out}')
    sys.exit(0)

# ---------------------------------------------------------------------------
# Load per-landscape popsize decay curves
# Shape after reshape: (10 popsizes, n_starts, 10 replicates, 25 generations)
# ---------------------------------------------------------------------------
popsizes = np.linspace(25, 2500, 10)

with open(os.path.join(slide_data_dir, 'decay_curves_GB1_m0.1_multistart_10000_uniform_multi_popsize.pkl'), 'rb') as f:
    GB1_popsize = pickle.load(f).reshape(10, -1, 10, 25)

with open(os.path.join(slide_data_dir, 'decay_curves_TrpB_m0.1_multistart_10000_uniform_multi_popsize.pkl'), 'rb') as f:
    TrpB_popsize = pickle.load(f).reshape(10, -1, 10, 25)

with open(os.path.join(slide_data_dir, 'decay_curves_TEV_m0.1_multistart_10000_uniform_multi_popsize.pkl'), 'rb') as f:
    TEV_popsize = pickle.load(f).reshape(10, -1, 10, 25)

with open(os.path.join(slide_data_dir, 'decay_curves_ParD3_m0.1_multistart_10000_uniform_multi_popsize.pkl'), 'rb') as f:
    ParD3_popsize = pickle.load(f).reshape(10, -1, 10, 25)

# ---------------------------------------------------------------------------
# array_results1: bootstrap rho over population sizes
# For each landscape, iterate popsizes[3:] (indices 3–9).
# Each bootstrap sample: draw array.shape[1]/16 start sites without
# replacement, average over sites AND replicates, then fit rho.
# Output shape per landscape: (7, n_sub)
# ---------------------------------------------------------------------------
eps   = 1e-8
n_sub = 250
np.random.seed(42)

array_results1 = []

for r, array in enumerate([GB1_popsize, TrpB_popsize, TEV_popsize, ParD3_popsize]):
    out1 = []

    for n in range(3, array.shape[0]):  # popsizes (indices 3–9)
        out2 = []

        for b in range(n_sub):  # bootstrap subsamples

            idx = np.random.choice(
                array.shape[1],
                size=int(array.shape[1] / 16),
                replace=False          # no duplicates within a subset
            )

            # mean over sites (idx) AND replicates, then square
            x = (array[n, idx, :, :].mean(axis=1) ** 2).mean(axis=0)

            x = x / x[0]

            rho = get_single_decay_rate(x)[0] / 2
            out2.append(rho)

        out1.append(out2)

    out1 = np.array(out1)   # shape: (7, n_sub)
    array_results1.append(out1)

# ---------------------------------------------------------------------------
# array_results2: bootstrap rho over number of generations sampled
# For each landscape, iterate g in range(2, 25).
# Each bootstrap sample: draw array.shape[1]/16 start sites without
# replacement from the highest-popsize slice, average over sites AND
# replicates for the first g generations, then fit rho.
# Output shape per landscape: (23, n_boot)
# ---------------------------------------------------------------------------
n_boot = 250
np.random.seed(42)

array_results2 = []

for array in [GB1_popsize, TrpB_popsize, TEV_popsize, ParD3_popsize]:

    out1 = []

    for g in range(2, array.shape[3]):   # generations (2–24)

        out2 = []

        for b in range(n_boot):  # bootstrap samples

            # bootstrap sampling (without replacement)
            idx = np.random.choice(
                array.shape[1],
                size=int(array.shape[1] / 16),
                replace=False
            )

            # take highest popsize, subset sites, keep all replicates
            x = (array[-1, idx, :, :g].mean(axis=1) ** 2).mean(axis=0)
            # mean over: sites (idx) + replicates

            x = x / x[0]

            rho = get_single_decay_rate(x, num_steps=g)[0] / 2
            out2.append(rho)

        out1.append(out2)

    out1 = np.array(out1)   # shape (23, n_boot)
    array_results2.append(out1)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
with open(out, 'wb') as f:
    pickle.dump((array_results1, array_results2), f)

print(f'Saved → {out}')
