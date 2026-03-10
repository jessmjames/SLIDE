"""
Preprocessing script: compute trajectory subsampling accuracy data.

Extracted from ruggedness_figures_data_processing.ipynb (cells 38, 53, 72-76).

For a given mutation model, loads the all-starts decay curves, computes
empirical heterogeneity, and bootstrap-subsamples to estimate how many
trajectories are needed to accurately estimate rho.

Usage
-----
  python compute_trajectory_subsampling.py                  # baseline (aa_uniform)
  python compute_trajectory_subsampling.py aa_uniform
  python compute_trajectory_subsampling.py nuc_symmetric
  python compute_trajectory_subsampling.py nuc_asymmetric
  python compute_trajectory_subsampling.py codon

Output
------
  plot_data/trajectory_subsampling_{model}.pkl

where model is the mutation model name (e.g. 'aa_uniform').
The baseline model also writes plot_data/trajectory_subsampling.pkl
for backward-compatibility with existing plot notebooks.
"""

import sys
import os
import pickle
import numpy as np
import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from direvo_functions import get_single_decay_rate_IK_v2
from slide_config import get_slide_data_dir

slide_data_dir = str(get_slide_data_dir())
plot_data_dir = os.path.join(parent_dir, 'plot_data')
os.makedirs(plot_data_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'aa_uniform'

# Filename suffix: baseline uses the old naming convention, others use the new one.
if MODEL == 'aa_uniform':
    suffix = 'all_starts'
else:
    suffix = f'{MODEL}_m0.1_all_starts'

LANDSCAPES = ['gb1', 'trpb', 'tev', 'pard3']

# Number of starting locations per landscape (for trajectory range).
# GB1 has 20^4 = 160 000, TrpB/TEV have 20^3 = 8 000, ParD3 (E3) has 20^3 = 8 000.
MAX_STARTS = {
    'gb1':   160_000,   # 20^4
    'trpb':  160_000,   # 20^4 — TrpB is 4-site
    'tev':   160_000,   # 20^4 — TEV is 4-site
    'pard3':   8_000,   # 20^3
}

N_BOOT = 1000
EPS    = 1e-8

# ---------------------------------------------------------------------------
# Load decay curves
# ---------------------------------------------------------------------------

print(f"Loading decay curves for model: {MODEL}")

decay = {}
for ld_name in LANDSCAPES:
    if MODEL == 'aa_uniform':
        fname = f"decay_curves_{ld_name}_m0.1_all_starts.pkl"
    else:
        fname = f"decay_curves_{ld_name}_{MODEL}_m0.1_all_starts.pkl"
    fpath = os.path.join(slide_data_dir, fname)
    with open(fpath, 'rb') as f:
        decay[ld_name] = pickle.load(f)
    print(f"  {ld_name}: {decay[ld_name].shape}")

# ---------------------------------------------------------------------------
# Compute empirical heterogeneity
# Shape of each decay array: (-1, 100, 10, 25)
# .mean(axis=2) → (-1, 100, 25)   (mean over reps)
# .reshape(-1, 25) → (total_starts, 25)
# ---------------------------------------------------------------------------

print("Computing empirical heterogeneity...")
heterogeneity = {
    ld_name: decay[ld_name].mean(axis=2).reshape(-1, 25)
    for ld_name in LANDSCAPES
}

# ---------------------------------------------------------------------------
# Bootstrap subsampling
# ---------------------------------------------------------------------------

print(f"Running bootstrap (n_boot={N_BOOT})...")
ld_results = []

for ld_name in LANDSCAPES:
    h = heterogeneity[ld_name]
    max_traj = MAX_STARTS[ld_name]

    trajectories = np.round(np.logspace(0, np.log10(max_traj), 11)).astype(int)
    traj_results = []

    for traj_number in tqdm.tqdm(trajectories, desc=ld_name):
        boot_vals = []

        for _ in range(N_BOOT):
            idx = np.random.choice(h.shape[0], size=int(traj_number), replace=True)
            sample = h[idx]
            sample = (sample ** 2).mean(axis=0)
            if sample[0] == 0:
                sample = sample / EPS
            else:
                sample = sample / sample[0]
            rho = get_single_decay_rate_IK_v2(sample)[0] / 2
            boot_vals.append(rho)

        traj_results.append(np.array(boot_vals))

    ld_results.append(traj_results)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

out_path = os.path.join(plot_data_dir, f"trajectory_subsampling_{MODEL}.pkl")
with open(out_path, 'wb') as f:
    pickle.dump(ld_results, f)
print(f"Saved → {out_path}")

# Backward-compatible copy for the baseline model.
if MODEL == 'aa_uniform':
    compat_path = os.path.join(plot_data_dir, 'trajectory_subsampling.pkl')
    with open(compat_path, 'wb') as f:
        pickle.dump(ld_results, f)
    print(f"Saved (compat) → {compat_path}")

print("Done.")
