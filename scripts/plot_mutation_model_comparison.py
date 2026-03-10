"""
Plot rho accuracy with increasing samples for multiple mutation models.

Loads trajectory_subsampling_{model}.pkl for each model and produces
a 2×2 subplot figure (one per landscape) showing how accuracy converges
with trajectory count under different mutation models.

Usage
-----
  python scripts/plot_mutation_model_comparison.py

Output
------
  figures/accuracy_over_sampling_mutation_models.pdf
"""

import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

plot_data_dir = os.path.join(parent_dir, 'plot_data')
figures_dir   = os.path.join(parent_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Models to compare: (label, pkl_suffix, linestyle)
# pkl_suffix matches the filename: trajectory_subsampling_{suffix}.pkl
# ---------------------------------------------------------------------------
MODELS = [
    ('AA uniform (baseline)', 'aa_uniform',      '-'),
    ('Nuc uniform',           'nuc_uniform',      '--'),
    ('Nuc symmetric',         'nuc_h_sapiens_sym','-.'),
    ('Nuc asymmetric',        'nuc_e_coli',       ':'),
]

LANDSCAPE_NAMES = ['GB1', 'TrpB', 'TEV', 'ParD3']
# Max starts per landscape (determines x-axis range)
MAX_STARTS = [160_000, 160_000, 160_000, 8_000]   # GB1/TrpB/TEV=20^4, ParD3=20^3

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
model_data = {}
for label, suffix, _ in MODELS:
    fpath = os.path.join(plot_data_dir, f'trajectory_subsampling_{suffix}.pkl')
    if not os.path.exists(fpath):
        print(f"  WARNING: {fpath} not found — skipping {label}")
        continue
    with open(fpath, 'rb') as f:
        model_data[suffix] = pickle.load(f)
    print(f"  Loaded {suffix}: {len(model_data[suffix])} landscapes")

if not model_data:
    raise FileNotFoundError("No trajectory_subsampling_*.pkl files found in plot_data/")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
colours = ['#4477AA', '#EE6677', '#228833', '#CCBB44']   # colorblind-friendly

fig, axes = plt.subplots(2, 2, figsize=(7, 6), dpi=300)
axes = axes.flatten()

for h, (ld_name, max_traj) in enumerate(zip(LANDSCAPE_NAMES, MAX_STARTS)):
    ax = axes[h]
    trajectories = np.round(np.logspace(0, np.log10(max_traj), 11)).astype(int)

    for label, suffix, ls in MODELS:
        if suffix not in model_data:
            continue
        ld_results = model_data[suffix]
        traj_results = ld_results[h]

        means = np.array([np.mean(traj_results[t]) for t in range(len(trajectories))])
        errs  = np.array([np.std(traj_results[t])  for t in range(len(trajectories))])

        ax.plot(trajectories, means, ls=ls, color=colours[h], label=label)
        ax.fill_between(trajectories, means - errs, means + errs,
                        alpha=0.12, color=colours[h], edgecolor=None)

    ax.set_xscale('log')
    ax.set_title(ld_name, fontsize=10)
    ax.set_xlabel('Trajectories averaged', fontsize=8)
    ax.set_ylabel(r'Decay rate $\rho$', fontsize=8)
    ax.tick_params(labelsize=7)

# Shared legend from the last axis
handles, labels = axes[-1].get_legend_handles_labels()
# Deduplicate (same label appears once per landscape; take first 4)
seen = {}
unique = [(h, l) for h, l in zip(handles, labels) if l not in seen and not seen.update({l: True})]
fig.legend([h for h, l in unique], [l for h, l in unique],
           loc='lower center', ncol=2, fontsize=8,
           bbox_to_anchor=(0.5, -0.04))

fig.suptitle(r'$\rho$ accuracy with increasing samples — mutation model comparison',
             fontsize=10, y=1.01)
plt.tight_layout()

out_path = os.path.join(figures_dir, 'accuracy_over_sampling_mutation_models.pdf')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"\nSaved → {out_path}")
plt.show()
