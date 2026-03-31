"""
Figure 4B — Step 2: compute rho distributions for NK and empirical landscapes.

Loads all-starts empirical decay curves and NK heterogeneity simulation data,
then fits decay rates (rho) for each starting point to produce distributions.

Input
-----
  SLIDE_data/decay_curves_gb1_m0.1_all_starts.pkl
  SLIDE_data/decay_curves_trpb_m0.1_all_starts.pkl
  SLIDE_data/decay_curves_tev_m0.1_all_starts.pkl
  SLIDE_data/decay_curves_pard3_m0.1_all_starts.pkl
  SLIDE_data/N4A20_heterogeneity.pkl

Output
------
  plot_data/heterogeneity_data.pkl
    (NK_rhos, empirical_rhos): lists of rho distributions per landscape

Note
----
  Originally from ruggedness_figures_data_processing.ipynb cells 49–57.
  NK_rhos uses 1000 random samples from N4A20_heterogeneity.pkl.
  empirical_rhos[0–2] use all 160000 trajectories; empirical_rhos[3]
  (ParD3) uses 8000 trajectories.
"""

import os
import sys
import pickle
import random
import numpy as np
import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

from ruggedness_functions import get_single_decay_rate, get_single_decay_rate_IK_v2
from slide_config import get_slide_data_dir

slide_data_dir = get_slide_data_dir()
plot_data_dir  = os.path.join(parent_dir, 'plot_data')
os.makedirs(plot_data_dir, exist_ok=True)

out = os.path.join(plot_data_dir, 'heterogeneity_data.pkl')

if os.path.exists(out):
    print(f'Already exists: {out}')
    sys.exit(0)

eps = 1e-8

# Load NK heterogeneity data
with open(os.path.join(slide_data_dir, 'N4A20_heterogeneity.pkl'), 'rb') as f:
    NK_heterogeneity = pickle.load(f)

# Load empirical all-starts decay curves
with open(os.path.join(slide_data_dir, 'decay_curves_gb1_m0.1_all_starts.pkl'), 'rb') as f:
    gb1_decay = pickle.load(f)
with open(os.path.join(slide_data_dir, 'decay_curves_trpb_m0.1_all_starts.pkl'), 'rb') as f:
    trpb_decay = pickle.load(f)
with open(os.path.join(slide_data_dir, 'decay_curves_tev_m0.1_all_starts.pkl'), 'rb') as f:
    tev_decay = pickle.load(f)
with open(os.path.join(slide_data_dir, 'decay_curves_pard3_m0.1_all_starts.pkl'), 'rb') as f:
    pard3_decay = pickle.load(f)

NK_heterogeneity = np.array([i.reshape(-1, 25) for i in NK_heterogeneity])
empirical_heterogeneity = [i.mean(axis=2).reshape(-1, 25)
                           for i in [gb1_decay, trpb_decay, tev_decay, pard3_decay]]

# NK rhos: 1000 random samples per K level
NK_rhos = [
    [get_single_decay_rate(NK_heterogeneity[ii][i] ** 2)[0] / 2
     for i in random.sample(range(1, 100001), 1000)]
    for ii in range(4)
]

# Empirical rhos: GB1, TrpB, TEV (160000 trajectories each)
empirical_rhos = []
for ii in range(3):
    rhos_ii = []
    for i in tqdm.tqdm(range(160000)):
        x = empirical_heterogeneity[ii][i]
        x = np.clip(x, eps, None)
        x = x ** 2
        if x[0] == 0:
            x = x / eps
        else:
            x = x / x[0]
        rho = get_single_decay_rate_IK_v2(x)[0] / 2
        rhos_ii.append(rho)
    empirical_rhos.append(rhos_ii)

# ParD3 (8000 trajectories)
rhos_ii = []
for i in tqdm.tqdm(range(8000)):
    x = pard3_decay.mean(axis=2).reshape(-1, 25)[i]
    x = np.clip(x, eps, None)
    x = x ** 2
    x = x / x[0]
    rho = get_single_decay_rate(x)[0] / 2
    rhos_ii.append(rho)
empirical_rhos.append(rhos_ii)

with open(out, 'wb') as f:
    pickle.dump((NK_rhos, empirical_rhos), f)

print(f'Saved → {out}')
