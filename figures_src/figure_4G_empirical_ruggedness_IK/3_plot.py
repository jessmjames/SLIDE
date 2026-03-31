"""
Figure 4G — Empirical Ruggedness Metric Comparison (IK)
Plotting script.

Source: ruggedness_figures_plots_IK.ipynb, cells 21-31.

Prerequisites:
  - pkl files from Jess's machine must be present in plot_data/:
      empirical_ruggedness_metric_comparison.pkl
      GB1_strategy_selection.pkl
      TrpB_strategy_selection.pkl
      TEV_strategy_selection.pkl
      E3_strategy_selection.pkl
  - Landscape arrays read from landscape_arrays/:
      GB1_landscape_array.pkl
      E3_landscape_array.pkl
      TEV_landscape_array.pkl
      TrpB_landscape_array.pkl
"""

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)
figures_dir = os.path.join(parent_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

import pickle
import numpy as np
import matplotlib.pyplot as plt
from ruggedness_functions import get_spectral_entropy, get_dirichlet_metric

# --- Shared settings (from cell 3 of IK notebook) ---
titlesize = 10
labelsize = 8
ticksize = 6
legendsize = 8
dpi = 350
plt.rcParams["font.family"] = "DejaVu Sans"
c2 = 'tab:blue'
c1 = 'tab:orange'
c3 = '#f55f74'
c4 = 'tab:green'

# --- Cell 21 ---
with open(os.path.join(parent_dir, 'plot_data', 'empirical_ruggedness_metric_comparison.pkl'), 'rb') as f:
    decay_rate_measurements, roughness_to_slope_measurements, landscape_r2_measurements, local_epistasis_measurements, paths_to_max_measurements, local_max_measurements = pickle.load(f)

# --- Cell 22 ---
with open(os.path.join(parent_dir, 'plot_data', 'GB1_strategy_selection.pkl'), 'rb') as f:
    _, _, gb1_decay_rate, gb1_sweep, _, _ = pickle.load(f)

with open(os.path.join(parent_dir, 'plot_data', 'TrpB_strategy_selection.pkl'), 'rb') as f:
    _, _, trpb_decay_rate, trpb_sweep, _, _ = pickle.load(f)

with open(os.path.join(parent_dir, 'plot_data', 'TEV_strategy_selection.pkl'), 'rb') as f:
    _, _, tev_decay_rate, tev_sweep, _, _ = pickle.load(f)

with open(os.path.join(parent_dir, 'plot_data', 'E3_strategy_selection.pkl'), 'rb') as f:
    _, _, pard3_decay_rate, pard3_sweep, _, _ = pickle.load(f)

# --- Cell 23 ---
with open(os.path.join(parent_dir, 'landscape_arrays', 'GB1_landscape_array.pkl'), 'rb') as f:
    GB1 = pickle.load(f)

with open(os.path.join(parent_dir, 'landscape_arrays', 'E3_landscape_array.pkl'), 'rb') as f:
    ParD3 = pickle.load(f)

with open(os.path.join(parent_dir, 'landscape_arrays', 'TEV_landscape_array.pkl'), 'rb') as f:
    TEV = pickle.load(f)

with open(os.path.join(parent_dir, 'landscape_arrays', 'TrpB_landscape_array.pkl'), 'rb') as f:
    TrpB = pickle.load(f)

# --- Cell 24 ---
from scipy.stats import percentileofscore

def normalise_array(x):
    x = np.asarray(x, dtype=float)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

evolvability = [normalise_array(sweep.mean(axis=(0,3)))[-1,0]/normalise_array(sweep.mean(axis=(0,3)))[0,-1] for sweep in [gb1_sweep, trpb_sweep, tev_sweep, pard3_sweep]]

# --- Cell 25 ---
decay_rate_measurements = [i[0]/2 for i in [gb1_decay_rate, trpb_decay_rate, tev_decay_rate, pard3_decay_rate]]

# --- Cell 26 ---
local_epistasis_measurements = [i['simple_sign_episasis']+ i['reciprocal_sign_epistasis'] for i in local_epistasis_measurements]

# --- Cell 27 ---
spectral_entropy_measurements = [get_spectral_entropy(l, remove_constant=True) for l in [GB1, TrpB, TEV, ParD3]]

# --- Cell 28 ---
dirichlet_energy_measurements = [get_dirichlet_metric(l) for l in [GB1, TrpB, TEV, ParD3]]

# --- Cell 31 ---
def r_sigfig(value, sigfig=1):
    if value == 0:
        return 0
    return np.round(value, -int(np.floor(np.log10(abs(value)))) + (sigfig - 1))

colours = [c1, c2, c4, c3]
labels = ['GB1', 'TrpB', 'TEV', 'ParD3']
rank_labels = ["1st", "2nd", "3rd", "4th"]
markers = ['o', 's', '^', 'D']
reversed_axes = [2, 5, 6]
titles = [r'Decay rate $\rho$', r'Norm. Dirichlet Energy $\rho_{\Delta}$', r'Landscape $R^2$', r'Spectral Entropy $H$', r'Local epistasis $n_{\epsilon}$', r'Dist. to local max $d_{max}$', r'Paths to max $n_{max}$', r'Roughness to slope $r/s$']
arr = np.array([
    decay_rate_measurements,
    np.array(dirichlet_energy_measurements),
    1-np.array(landscape_r2_measurements),
    np.array(spectral_entropy_measurements),
    local_epistasis_measurements,
    np.array(local_max_measurements),
    np.array(paths_to_max_measurements),
    roughness_to_slope_measurements
])
norm_axes = [0,1,2,3]

fig, axes = plt.subplots(1, len(arr), figsize=(2+len(arr), 2), dpi=300)

for i, ax in enumerate(axes):
    y_positions = arr[i]
    x_positions = np.full(4, 0.5)

    order = np.argsort(y_positions) if i in reversed_axes else np.argsort(-y_positions)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(y_positions))

    for j in range(4):
        jitter = np.random.uniform(-0.1, 0.1)
        jitter = 0.05 if j % 2 == 0 else -0.05
        x = x_positions[j] + jitter
        y = y_positions[j]

        ax.scatter(x, y, marker=markers[j],
                   color=colours[j], s=40, edgecolors='black', zorder=3,
                   label=labels[j] if i == 0 else None)
        ax.text(x + 0.1 if j % 2 == 0 else x - 0.1, y, rank_labels[ranks[j]],
                fontsize=5, va='center', ha='left' if j % 2 == 0 else 'right', zorder=4)

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    ax.set_xticks([])
    if i in norm_axes:
        print(f"norm axes {titles[i]}")
        ax.set_ylim(0, 1)
        ax.set_yticks([0,1])
        ax.set_yticklabels([0,1], fontsize=5)
    else:
        ax.set_yticks([arr[i].min(), arr[i].max()])
        ax.set_yticklabels([r_sigfig(arr[i].min()), r_sigfig(arr[i].max())], fontsize=5)
    ax.set_xlim(0, 1)
    ax.margins(y=0.1)
    ax.set_title(titles[i], fontsize=titlesize-4.5)

axes[0].legend(fontsize=legendsize, loc='upper left', bbox_to_anchor=(0.7+3+len(arr), 0.8))

for idx in reversed_axes:
    axes[idx].invert_yaxis()

plt.subplots_adjust(wspace=0.5)
fig.text(0.08, 0.5, r'Ruggedness (a.u.) $\rightarrow$', fontsize=8, va='center', rotation=90)

plt.savefig(os.path.join(figures_dir, 'empirical_ruggedness_metric_comparison_IK.pdf'), dpi=dpi)
print(f'Saved → {os.path.join(figures_dir, "empirical_ruggedness_metric_comparison_IK.pdf")}')
