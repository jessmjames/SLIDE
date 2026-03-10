"""
Cross-reference plot: spectral rho vs simulation-based rho at convergence.

Loads:
  plot_data/spectral_rho_comparison.pkl          (from compute_spectral_rho_models.py)
  plot_data/trajectory_subsampling_{model}.pkl   (from compute_trajectory_subsampling.py)

The simulation "true rho" is taken as the mean of the bootstrap distribution at
the highest trajectory count (i.e. the full-data estimate).

Outputs:
  figures/spectral_vs_simulation_rho.pdf   — scatter plot (spectral vs simulation)
  figures/spectral_vs_simulation_rho_bars.pdf — bar chart per landscape/model
"""

import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import pickle
import numpy as np
import matplotlib.pyplot as plt

plot_data_dir = os.path.join(parent_dir, 'plot_data')
figures_dir   = os.path.join(parent_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

LANDSCAPE_NAMES = ['GB1', 'TrpB', 'TEV', 'ParD3']

# Models: (display label, spectral key, subsampling pkl suffix, spectral_valid)
# spectral_valid=False means T is asymmetric — Dirichlet form not well-defined;
# spectral rho plotted as NaN / omitted from scatter.
MODELS = [
    ('AA uniform',    'aa_uniform',        'aa_uniform',        True),
    ('Nuc uniform',   'nuc_uniform',        'nuc_uniform',       True),
    ('Nuc symmetric', 'nuc_h_sapiens_sym',  'nuc_h_sapiens_sym', True),
    ('Nuc E.coli',    'nuc_e_coli',         'nuc_e_coli',        False),
]

# ---------------------------------------------------------------------------
# Load spectral rho
# ---------------------------------------------------------------------------

with open(os.path.join(plot_data_dir, 'spectral_rho_comparison.pkl'), 'rb') as f:
    spectral = pickle.load(f)

# ---------------------------------------------------------------------------
# Load simulation rho at convergence (mean at highest trajectory count)
# ---------------------------------------------------------------------------

sim_rho = {}   # {model_suffix: [rho_GB1, rho_TrpB, rho_TEV, rho_ParD3]}

for _, _, suffix, _ in MODELS:
    fpath = os.path.join(plot_data_dir, f'trajectory_subsampling_{suffix}.pkl')
    if not os.path.exists(fpath):
        print(f"  WARNING: {fpath} not found — skipping {suffix}")
        sim_rho[suffix] = [float('nan')] * len(LANDSCAPE_NAMES)
        continue
    with open(fpath, 'rb') as f:
        data = pickle.load(f)  # list of landscapes, each list of trajectory counts
    # data[landscape_idx][traj_count_idx] = array of N_BOOT rho values
    # Take last trajectory count (highest = full convergence)
    vals = [float(np.mean(data[i][-1])) for i in range(len(LANDSCAPE_NAMES))]
    sim_rho[suffix] = vals
    print(f"  {suffix}: sim rho = {vals}")

# ---------------------------------------------------------------------------
# Assemble into arrays for plotting
# ---------------------------------------------------------------------------

n_ld    = len(LANDSCAPE_NAMES)
n_model = len(MODELS)

spec_arr = np.zeros((n_model, n_ld))
sim_arr  = np.zeros((n_model, n_ld))

for mi, (label, spec_key, suffix, is_sym) in enumerate(MODELS):
    for li, ld_name in enumerate(LANDSCAPE_NAMES):
        spec_arr[mi, li] = spectral[ld_name].get(spec_key, float('nan'))
        sim_arr[mi, li]  = sim_rho[suffix][li]

# ---------------------------------------------------------------------------
# Figure 1: scatter plot (spectral vs simulation)
# ---------------------------------------------------------------------------

colours   = ['#4477AA', '#EE6677', '#228833', '#CCBB44']
markers   = ['o', 's', '^', 'D']
ld_colours = ['#4477AA', '#EE6677', '#228833', '#CCBB44']

fig, ax = plt.subplots(figsize=(5, 5), dpi=150)

all_vals = np.concatenate([spec_arr.flatten(), sim_arr.flatten()])
all_vals = all_vals[np.isfinite(all_vals)]
vmin, vmax = 0, max(all_vals) * 1.05

for mi, (label, _, _, _) in enumerate(MODELS):
    for li, ld_name in enumerate(LANDSCAPE_NAMES):
        x = spec_arr[mi, li]
        y = sim_arr[mi, li]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        ax.scatter(x, y, color=ld_colours[li], marker=markers[mi],
                   s=80, zorder=3, label=f'{label} / {ld_name}')

# Identity line
ax.plot([vmin, vmax], [vmin, vmax], 'k--', lw=0.8, alpha=0.5, label='y = x')

# Build a clean legend: models (markers) + landscapes (colours)
from matplotlib.lines import Line2D
model_handles = [Line2D([0], [0], marker=markers[mi], color='grey', linestyle='none',
                         markersize=7, label=label)
                 for mi, (label, _, _, _) in enumerate(MODELS)]
ld_handles    = [Line2D([0], [0], marker='o', color=ld_colours[li], linestyle='none',
                         markersize=7, label=LANDSCAPE_NAMES[li])
                 for li in range(n_ld)]
ax.legend(handles=model_handles + ld_handles, fontsize=7,
          loc='upper left', framealpha=0.8)

ax.set_xlim(vmin, vmax)
ax.set_ylim(vmin, vmax)
ax.set_xlabel('Spectral rho (analytic)', fontsize=10)
ax.set_ylabel('Simulation rho (convergence)', fontsize=10)
ax.set_title('Spectral vs simulation-based rho', fontsize=10)
ax.tick_params(labelsize=8)

plt.tight_layout()
out1 = os.path.join(figures_dir, 'spectral_vs_simulation_rho.pdf')
plt.savefig(out1, bbox_inches='tight')
print(f'Saved → {out1}')
plt.close()

# ---------------------------------------------------------------------------
# Figure 2: grouped bar chart
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, n_ld, figsize=(10, 3.5), dpi=150, sharey=False)
bar_width = 0.35
x = np.arange(n_model)

for li, (ld_name, ax) in enumerate(zip(LANDSCAPE_NAMES, axes)):
    b1 = ax.bar(x - bar_width/2, spec_arr[:, li], bar_width,
                color=ld_colours[li], alpha=0.85, label='Spectral')
    b2 = ax.bar(x + bar_width/2, sim_arr[:, li],  bar_width,
                color=ld_colours[li], alpha=0.40, hatch='//', label='Simulation')
    ax.set_title(ld_name, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in MODELS], rotation=30, ha='right', fontsize=7)
    ax.set_ylabel(r'$\rho$', fontsize=9)
    ax.tick_params(labelsize=7)
    if li == 0:
        ax.legend(fontsize=7)

fig.suptitle('Spectral rho (solid) vs simulation rho at convergence (hatched)',
             fontsize=9, y=1.01)
plt.tight_layout()
out2 = os.path.join(figures_dir, 'spectral_vs_simulation_rho_bars.pdf')
plt.savefig(out2, bbox_inches='tight')
print(f'Saved → {out2}')
plt.close()
