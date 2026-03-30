"""
Figure S4 — Step 3: plot strategy space heatmaps per generation.

For each generation count in [25, 50, 100], produces a 1×3 figure
(one panel per K value) showing the strategy space heatmap with the
optimal strategy marked.

Input
-----
  plot_data/strategy_gens_data.pkl

Output
------
  figures/strategy_prediction_25gens.pdf
  figures/strategy_prediction_50gens.pdf
  figures/strategy_prediction_100gens.pdf

Usage
-----
  python figures_src/figure_S4_strategy_gens/3_plot.py

Originally from ruggedness_figures_data_processing_IK.ipynb cells 83–89.
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

with open(os.path.join(plot_data_dir, 'strategy_gens_data.pkl'), 'rb') as f:
    d = pickle.load(f)

all_strategy_data = d['all_strategy_data']   # (5, 3, 7, 7)
gens              = d['gens']                # [5, 25, 50, 75, 100]
splits            = d['splits']              # [24, 20, 16, 12, 8, 4, 1]
base_chances      = d['base_chances']

K_labels = ['K=0', 'K=1', 'K=2']

for g in [25, 50, 100]:
    gen_idx = gens.index(g)
    strategy_data = all_strategy_data[gen_idx]   # (3, 7, 7)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=300)
    fig.suptitle(f'Strategy space — {g} generations', fontsize=11)

    for j, ax in enumerate(axes):
        data    = strategy_data[j]
        im      = ax.imshow(data, origin='lower', cmap='viridis', aspect='auto')
        max_idx = np.unravel_index(np.argmax(data), data.shape)
        y_max, x_max = max_idx
        ax.scatter(x_max, y_max, color='red', s=25, marker='o',
                   edgecolors='white', linewidth=0.5, zorder=3)

        ax.set_xticks(range(7))
        ax.set_yticks(range(7))
        ax.set_xticklabels([f'{bc:.2f}' for bc in base_chances], fontsize=6, rotation=45)
        ax.set_yticklabels(splits, fontsize=6)
        ax.set_title(K_labels[j], fontsize=9)
        ax.set_xlabel('Base chance', fontsize=8)
        if j == 0:
            ax.set_ylabel('Population split', fontsize=8)

        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    out = os.path.join(figures_dir, f'strategy_prediction_{g}gens.pdf')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved → {out}')
    plt.close()
