"""
Accuracy-over-sampling plot for ParD3 AA uniform only.
Compares replace=True vs replace=False bootstrap.
"""

import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import pickle
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from direvo_functions import get_single_decay_rate_IK_v2
from ruggedness_functions import get_dirichlet_metric
from slide_config import get_slide_data_dir

slide_data_dir = str(get_slide_data_dir())
figures_dir    = os.path.join(parent_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

# Load ParD3 AA uniform decay curves
fpath = os.path.join(slide_data_dir, 'decay_curves_pard3_m0.1_all_starts.pkl')
with open(fpath, 'rb') as f:
    data = pickle.load(f)
print(f"Loaded decay curves: {data.shape}")  # (-1, 100, 10, 25)

# Load spectral reference
import pickle as pkl
with open(os.path.join(parent_dir, 'plot_data', 'spectral_rho_comparison.pkl'), 'rb') as f:
    spectral_rho_all = pkl.load(f)
spectral_rho = spectral_rho_all['ParD3']['aa_uniform']
print(f"Spectral rho: {spectral_rho:.4f}")

# h[i, t] = mean over K seeds of mean population fitness
# data shape: (-1, 100, 10, 25) -> reshape to (total_starts, 10, 25)
h = data.mean(axis=2).reshape(-1, 25)   # (total_starts, 25)  — mean over 10 seeds
total_starts = h.shape[0]
print(f"Total starting points: {total_starts}")

# Trajectory counts (x-axis)
trajectories = np.round(np.logspace(0, np.log10(8000), 11)).astype(int)
N_BOOT = 1000
EPS = 1e-8

def run_bootstrap(replace):
    rng = np.random.default_rng(0)
    traj_results = []
    for traj_number in tqdm.tqdm(trajectories, desc=f'replace={replace}'):
        boot_vals = []
        for _ in range(N_BOOT):
            idx    = rng.choice(total_starts, size=int(traj_number), replace=replace)
            sample = (h[idx] ** 2).mean(axis=0)
            sample = sample / (np.abs(sample[0]) + EPS)
            rho    = get_single_decay_rate_IK_v2(sample)[0] / 2
            boot_vals.append(rho)
        traj_results.append(np.array(boot_vals))
    return traj_results

results_with    = run_bootstrap(replace=True)
results_without = run_bootstrap(replace=False)

# Plot
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

for results, label, color in [
    (results_with,    'replace=True',  '#4477AA'),
    (results_without, 'replace=False', '#EE6677'),
]:
    means = np.array([r.mean() for r in results])
    stds  = np.array([r.std()  for r in results])
    ax.plot(trajectories, means, color=color, label=label)
    ax.fill_between(trajectories, means - stds, means + stds,
                    alpha=0.15, color=color, edgecolor=None)

ax.axhline(spectral_rho, ls='--', color='k', lw=1.2, label=f'Spectral rho = {spectral_rho:.3f}')
ax.set_xscale('log')
ax.set_xlabel('Starting points sampled')
ax.set_ylabel(r'IK $\rho$ estimate')
ax.set_title('ParD3 AA uniform — accuracy vs sampling')
ax.legend(fontsize=9)
plt.tight_layout()

out = os.path.join(figures_dir, 'pard3_aa_sampling_replace.pdf')
plt.savefig(out, bbox_inches='tight')
print(f"Saved → {out}")
