"""
Figure 5A — Step 3: plot predicted DE strategies vs true rho.

Input
-----
  plot_data/optimal_DE_strategies.pkl
    (decay_rates, optimal_splits, optimal_base_chances)

  plot_data/strategy_prediction_accuracy.pkl
    (actual_k_over_ns, bc_means, bc_stds, sp_means, sp_stds)

Output
------
  figures/strategy_prediction.pdf

Usage
-----
  python figures_src/figure_5A_strategy_prediction/3_plot.py

Originally from ruggedness_figures_plots.ipynb cell 46.
"""

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

import pickle
import numpy as np
import matplotlib.pyplot as plt

plot_data_dir = os.path.join(parent_dir, 'plot_data')
figures_dir   = os.path.join(parent_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

with open(os.path.join(plot_data_dir, 'optimal_DE_strategies.pkl'), 'rb') as f:
    decay_rates, optimal_splits, optimal_base_chances = pickle.load(f)

with open(os.path.join(plot_data_dir, 'strategy_prediction_accuracy.pkl'), 'rb') as f:
    actual_k_over_ns, bc_means, bc_stds, sp_means, sp_stds = pickle.load(f)

c1 = 'tab:orange'
c2 = 'tab:blue'
legendsize = 8
dpi = 350

fig, ax1 = plt.subplots(figsize=(3.5,3), dpi=300)

k_over_ns = np.unique(np.round(actual_k_over_ns,1))

color = c1
ax1.set_xlabel(r'True $\rho$', fontsize=8)
ax1.set_ylabel(r'Predicted base chance $b$', color=color, fontsize=8)
ax1.errorbar(k_over_ns, bc_means, yerr=bc_stds, fmt='o', capsize=5, color=c1)
ax1.plot(decay_rates, optimal_base_chances, color=color, linestyle='--', alpha=0.6, label='Optimal base chance')
ax1.tick_params(axis='y', labelcolor=color, labelsize=6)
ax1.tick_params(axis='x', labelsize=6)
ax1.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.3)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = c2
ax2.set_ylabel('Predicted splitting (total 1200)', color=color, fontsize=8)
ax2.errorbar(k_over_ns, sp_means, yerr=sp_stds, fmt='o', capsize=5, color=c2)
ax2.plot(decay_rates, optimal_splits, color=color, linestyle='--', alpha=0.6, label='Optimal splitting')
ax2.tick_params(axis='y', labelcolor=color, labelsize=6)

ax1.set_title(r'Predicted DE strategies vs true $\rho$', fontsize=10)

# Adjust legend positions
ax1.legend(fontsize=legendsize-2, loc='upper left', bbox_to_anchor=(0.0, 1.0))
ax2.legend(fontsize=legendsize-2, loc='upper left', bbox_to_anchor=(0.0, 0.9))  # Move slightly below ax1's legend

plt.xlim(0.05, 1.05)
fig.tight_layout()

out_path = os.path.join(figures_dir, 'strategy_prediction.pdf')
plt.savefig(out_path, dpi=dpi)
print(f'Saved -> {out_path}')
