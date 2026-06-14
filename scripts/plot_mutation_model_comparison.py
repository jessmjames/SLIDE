"""
Plot rho accuracy with increasing samples for nuc mutation models.

Three subplots (uniform, symmetric, asymmetric mutation), one colour per landscape.
AA uniform is saved as a separate single-panel figure.

Usage
-----
  python scripts/plot_mutation_model_comparison.py [--steps N]

Output
------
  figures/graph_mutation_model.pdf
  figures/accuracy_over_sampling_aa_uniform.pdf
"""

import sys
import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

plot_data_dir = os.path.join(parent_dir, 'plot_data')
figures_dir   = os.path.join(parent_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

NUM_STEPS    = 25
if '--steps' in sys.argv:
    NUM_STEPS = int(sys.argv[sys.argv.index('--steps') + 1])
STEPS_SUFFIX = f'_{NUM_STEPS}steps' if NUM_STEPS != 25 else ''

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

with open(os.path.join(plot_data_dir, 'spectral_rho_comparison.pkl'), 'rb') as f:
    spectral_rho = pickle.load(f)

LANDSCAPE_NAMES = ['GB1', 'TrpB', 'TEV', 'ParD3']
MAX_STARTS      = [160_000, 160_000, 160_000, 8_000]

# GB1=orange, TrpB=blue, TEV=green, ParD3=red  (consistent with decay curves plot)
colours = [matplotlib.colormaps['tab10'](i) for i in [1, 0, 2, 3]]

# Nuc models (main figure)
NUC_MODELS = [
    ('Uniform mutation',            'nuc_uniform',       'nuc_uniform'),
    ('H. sapiens (symmetric)',      'nuc_h_sapiens_sym', 'nuc_h_sapiens_sym'),
    ('E. coli (asymmetric)',        'nuc_e_coli',        'nuc_e_coli_sym'),
]

# AA uniform (separate figure)
AA_MODEL = ('AA uniform', 'aa_uniform', 'aa_uniform')


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def load_model(suffix):
    fpath = os.path.join(plot_data_dir, f'trajectory_subsampling_{suffix}{STEPS_SUFFIX}.pkl')
    if not os.path.exists(fpath):
        print(f'  WARNING: {fpath} not found')
        return None
    with open(fpath, 'rb') as f:
        return pickle.load(f)


def draw_panel(ax, suffix, spec_key, title):
    data = load_model(suffix)
    if data is None:
        ax.set_visible(False)
        return

    for h, (ld_name, max_traj) in enumerate(zip(LANDSCAPE_NAMES, MAX_STARTS)):
        trajectories = np.round(np.logspace(0, np.log10(max_traj), 11)).astype(int)
        traj_results = data[h]

        means = np.array([np.mean(traj_results[t]) for t in range(len(trajectories))])
        errs  = np.array([np.std(traj_results[t])  for t in range(len(trajectories))])

        ax.plot(trajectories, means, color=colours[h], lw=1.5, label=ld_name)
        ax.fill_between(trajectories, means - errs, means + errs,
                        alpha=0.12, color=colours[h], edgecolor=None)

        if spec_key:
            ref = spectral_rho.get(ld_name, {}).get(spec_key, float('nan'))
            if not np.isnan(ref):
                ax.axhline(ref, ls=':', color=colours[h], alpha=0.6, lw=1.0)

    ax.set_xscale('log')
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('Starting points used', fontsize=8)
    ax.set_ylabel(r'Estimated $\rho$', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ---------------------------------------------------------------------------
# Main figure: 3 nuc models
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), dpi=300)

for ax, (title, suffix, spec_key) in zip(axes, NUC_MODELS):
    draw_panel(ax, suffix, spec_key, title)

# Legend: landscape colours + a single "true rho" dotted-line entry
landscape_handles = [
    mlines.Line2D([], [], color=colours[h], lw=1.5, label=name)
    for h, name in enumerate(LANDSCAPE_NAMES)
]
true_rho_handle = mlines.Line2D([], [], color='grey', ls=':', lw=1.0, label=r'True $\rho$')
fig.legend(handles=landscape_handles + [true_rho_handle],
           loc='lower center', ncol=5, fontsize=7.5,
           bbox_to_anchor=(0.5, -0.12))

plt.tight_layout()
out_path = os.path.join(figures_dir, f'graph_mutation_model{STEPS_SUFFIX}.pdf')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f'Saved → {out_path}')

# ---------------------------------------------------------------------------
# Composite-ready: 3 individual panel PDFs + legend strip
# Each panel is exactly graph_mutation_model width / 3 = 193.22 pts = 2.684 in
# ---------------------------------------------------------------------------

# Panels for LaTeX composite (no d/e/f labels — LaTeX adds those)
PANEL_W_IN  = 3.0
PANEL_H_IN  = 2.5

for (title, suffix, spec_key), label in zip(NUC_MODELS, ['d', 'e', 'f']):
    figp, axp = plt.subplots(1, 1, figsize=(PANEL_W_IN, PANEL_H_IN))
    figp.subplots_adjust(left=0.17, right=0.97, top=0.90, bottom=0.20)
    draw_panel(axp, suffix, spec_key, title)
    panel_path = os.path.join(figures_dir, f'accuracy_panel_{label}{STEPS_SUFFIX}.pdf')
    figp.savefig(panel_path, dpi=300)
    plt.close(figp)
    print(f'Saved panel → {panel_path}')

# Legend strip (full width, for LaTeX to include below panels)
fig_leg, ax_leg = plt.subplots(1, 1, figsize=(8.05, 0.45))
ax_leg.set_visible(False)
landscape_handles_leg = [
    mlines.Line2D([], [], color=colours[h], lw=1.5, label=name)
    for h, name in enumerate(LANDSCAPE_NAMES)
]
true_rho_handle_leg = mlines.Line2D([], [], color='grey', ls=':', lw=1.0, label=r'True $\rho$')
fig_leg.legend(handles=landscape_handles_leg + [true_rho_handle_leg],
               loc='center', ncol=5, fontsize=7.5, frameon=False)
legend_path = os.path.join(figures_dir, f'accuracy_panel_legend{STEPS_SUFFIX}.pdf')
fig_leg.savefig(legend_path, dpi=300)
plt.close(fig_leg)
print(f'Saved legend → {legend_path}')

# ---------------------------------------------------------------------------
# Separate figure: AA uniform
# ---------------------------------------------------------------------------

fig2, ax2 = plt.subplots(1, 1, figsize=(3.5, 3.2), dpi=300)
draw_panel(ax2, AA_MODEL[1], AA_MODEL[2], AA_MODEL[0])

landscape_handles2 = [
    mlines.Line2D([], [], color=colours[h], lw=1.5, label=name)
    for h, name in enumerate(LANDSCAPE_NAMES)
]
true_rho_handle2 = mlines.Line2D([], [], color='grey', ls=':', lw=1.0, label=r'True $\rho$')
ax2.legend(handles=landscape_handles2 + [true_rho_handle2], fontsize=7.5, frameon=False)

plt.tight_layout()
out_path2 = os.path.join(figures_dir, f'accuracy_over_sampling_aa_uniform{STEPS_SUFFIX}.pdf')
plt.savefig(out_path2, dpi=300, bbox_inches='tight')
print(f'Saved → {out_path2}')
