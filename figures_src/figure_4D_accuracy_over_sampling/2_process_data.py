"""
Figure 4D — Step 2: bootstrap rho estimates across trajectory sub-sample sizes.

Loads empirical all-starts decay curves and computes bootstrapped rho
estimates at 11 log-spaced trajectory counts for each landscape.

Input
-----
  SLIDE_data/decay_curves_gb1_m0.1_all_starts.pkl
  SLIDE_data/decay_curves_trpb_m0.1_all_starts.pkl
  SLIDE_data/decay_curves_tev_m0.1_all_starts.pkl
  SLIDE_data/decay_curves_pard3_m0.1_all_starts.pkl

Output
------
  plot_data/trajectory_subsampling.pkl
    ld_results: list of 4 elements (one per landscape: GB1, TrpB, TEV, ParD3).
    Each element is a list of 11 arrays, one per trajectory count, each
    containing 1000 bootstrapped rho estimates.

Note
----
  Originally from ruggedness_figures_data_processing.ipynb cells 71–75.
  GB1/TrpB/TEV use trajectories up to 160000; ParD3 uses up to 8000.
  n_boot = 1000 bootstrap resamples per trajectory count.
"""

import os
import sys
import pickle
import numpy as np
import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

from ruggedness_functions import get_single_decay_rate_IK_v2
from slide_config import get_slide_data_dir

slide_data_dir = get_slide_data_dir()
plot_data_dir  = os.path.join(parent_dir, 'plot_data')
os.makedirs(plot_data_dir, exist_ok=True)

out = os.path.join(plot_data_dir, 'trajectory_subsampling.pkl')

if os.path.exists(out):
    print(f'Already exists: {out}')
    sys.exit(0)

eps = 1e-8

with open(os.path.join(slide_data_dir, 'decay_curves_gb1_m0.1_all_starts.pkl'), 'rb') as f:
    gb1_decay = pickle.load(f)
with open(os.path.join(slide_data_dir, 'decay_curves_trpb_m0.1_all_starts.pkl'), 'rb') as f:
    trpb_decay = pickle.load(f)
with open(os.path.join(slide_data_dir, 'decay_curves_tev_m0.1_all_starts.pkl'), 'rb') as f:
    tev_decay = pickle.load(f)
with open(os.path.join(slide_data_dir, 'decay_curves_pard3_m0.1_all_starts.pkl'), 'rb') as f:
    pard3_decay = pickle.load(f)

empirical_heterogeneity = [i.mean(axis=2).reshape(-1, 25)
                           for i in [gb1_decay, trpb_decay, tev_decay, pard3_decay]]

trajectories = np.round(np.logspace(0, np.log10(160000), 11)).astype(int)

n_boot = 1000
ld_results = []

for i in empirical_heterogeneity[:3]:

    traj_results = []   # one entry per traj_number

    for traj_number in trajectories:

        boot_vals = []  # 100 bootstrap samples for this traj_number

        for _ in range(n_boot):

            idx = np.random.choice(
                i.shape[0],
                size=int(traj_number),
                replace=True
            )

            sample = i[idx]
            sample = (sample**2).mean(axis=0)
            if sample[0] == 0:
                sample = sample/eps
            else:
                sample = sample / sample[0]
            sample = get_single_decay_rate_IK_v2(sample)[0] / 2

            boot_vals.append(sample)

        traj_results.append(np.array(boot_vals))

    ld_results.append(traj_results)

## Adding ParD3

trajectories = np.round(np.logspace(0, np.log10(8000), 11)).astype(int)

i = empirical_heterogeneity[-1]

traj_results = []   # one entry per traj_number

for traj_number in trajectories:

    boot_vals = []  # 100 bootstrap samples for this traj_number

    for _ in range(n_boot):

        idx = np.random.choice(
            i.shape[0],
            size=int(traj_number),
            replace=True
        )

        sample = i[idx]
        sample = (sample**2).mean(axis=0)
        if sample[0] == 0:
            sample = sample/eps
        else:
            sample = sample / sample[0]
        sample = get_single_decay_rate_IK_v2(sample)[0] / 2

        boot_vals.append(sample)

    traj_results.append(np.array(boot_vals))

ld_results.append(traj_results)

with open(out, 'wb') as f:
    pickle.dump(ld_results, f)

print(f'Saved → {out}')
