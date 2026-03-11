"""
Plot rho accuracy with increasing samples for multiple mutation models.

One subplot per mutation model, one colour per landscape.

Usage
-----
  python scripts/plot_mutation_model_comparison.py

Output
------
  figures/accuracy_over_sampling_mutation_models_v2.pdf
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

# Load spectral reference values
spectral_path = os.path.join(plot_data_dir, 'spectral_rho_comparison.pkl')
with open(spectral_path, 'rb') as f:
    spectral_rho = pickle.load(f)

SPECTRAL_LD_KEYS = ['GB1', 'TrpB', 'TEV', 'ParD3']
SPECTRAL_MODEL_KEYS = {
    'aa_uniform':        'aa_uniform',
    'nuc_uniform':       'nuc_uniform',
    'nuc_h_sapiens_sym': 'nuc_h_sapiens_sym',
    'nuc_e_coli':        None,   # asymmetric — no spectral ref
}

MODELS = [
    ('AA uniform (baseline)', 'aa_uniform',       '-'),
    ('Nuc uniform',           'nuc_uniform',       '-'),
    ('Nuc symmetric',         'nuc_h_sapiens_sym', '-'),
    ('Nuc asymmetric',        'nuc_e_coli',        '-'),
]

LANDSCAPE_NAMES = ['GB1', 'TrpB', 'TEV', 'ParD3']
MAX_STARTS = [160_000, 160_000, 160_000, 8_000]

# Colours: one per landscape
colours = ['#4477AA', '#EE6677', '#228833', '#CCBB44']

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
# Plot: one subplot per mutation model
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(7, 6), dpi=300)
axes = axes.flatten()

for mi, (model_label, suffix, _) in enumerate(MODELS):
    ax = axes[mi]

    if suffix not in model_data:
        ax.set_visible(False)
        continue

    spec_model_key = SPECTRAL_MODEL_KEYS.get(suffix)

    for h, (ld_name, max_traj) in enumerate(zip(LANDSCAPE_NAMES, MAX_STARTS)):
        trajectories = np.round(np.logspace(0, np.log10(max_traj), 11)).astype(int)
        ld_results = model_data[suffix]
        traj_results = ld_results[h]

        means = np.array([np.mean(traj_results[t]) for t in range(len(trajectories))])
        errs  = np.array([np.std(traj_results[t])  for t in range(len(trajectories))])

        ax.plot(trajectories, means, color=colours[h], label=ld_name)
        ax.fill_between(trajectories, means - errs, means + errs,
                        alpha=0.12, color=colours[h], edgecolor=None)

        # Spectral reference line (dashed, same landscape colour)
        if spec_model_key:
            ld_key = SPECTRAL_LD_KEYS[h]
            if ld_key in spectral_rho:
                ref = spectral_rho[ld_key].get(spec_model_key, float('nan'))
                if not np.isnan(ref):
                    ax.axhline(ref, ls='--', color=colours[h], alpha=0.5, lw=1.0)

    ax.set_xscale('log')
    ax.set_title(model_label, fontsize=10)
    ax.set_xlabel('Trajectories averaged', fontsize=8)
    ax.set_ylabel(r'Decay rate $\rho$', fontsize=8)
    ax.tick_params(labelsize=7)

# Shared legend (landscapes) from first axis
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=8,
           bbox_to_anchor=(0.5, -0.04))

fig.suptitle(r'$\rho$ accuracy with increasing samples — mutation model comparison',
             fontsize=10, y=1.01)
plt.tight_layout()

out_path = os.path.join(figures_dir, 'accuracy_over_sampling_mutation_models_v2.pdf')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"\nSaved → {out_path}")
