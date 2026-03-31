"""
Figure 5G — Step 3: plot ParD3 directed evolution (decay curve, strategy space).

Input
-----
  plot_data/ParD3_strategy_selection.pkl
    (x_vals, decay_mean, decay_rate, sweep, scipy_freq_matrix, run, scatter, line,
     ParD3_decay_multi)

Output
------
  figures/ParD3_decay.pdf
  figures/ParD3_strategy_space.pdf

Note on cell 77 (copy-paste artefact):
  Cell 77 in the notebook runs the DE plotting code on the ParD3 `run` variable
  but saves to 'figures/TEV_DE.pdf' and uses the title
  'TrpB directed evolution, 10 start average'. This is an apparent copy-paste
  error in the original notebook. It is NOT reproduced here because Fig 5G only
  covers the decay curve and strategy space panels; the TEV_DE.pdf output
  properly belongs to Fig 5F. Cell 77 is preserved faithfully in the notebook
  but omitted from this script to avoid overwriting the TEV figure with ParD3 data.

Usage
-----
  python figures_src/figure_5G_ParD3_DE/3_plot.py

Originally from ruggedness_figures_plots.ipynb cells 75, 76.
Cell 77 is intentionally excluded — see note above.
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

with open(os.path.join(plot_data_dir, 'ParD3_strategy_selection.pkl'), 'rb') as f:
    x_vals, decay_mean, decay_rate, sweep, scipy_freq_matrix, run, scatter, line, ParD3_decay_multi = pickle.load(f)

c1 = 'tab:orange'
c2 = 'tab:blue'
titlesize = 10
labelsize = 8
ticksize = 6
legendsize = 8
dpi = 350

# --- Cell 75: ParD3 decay curve ---
plt.figure(figsize=(3, 1.5), dpi=300)
plt.scatter(x_vals, scatter, label='Mean fitness', s=15)
plt.plot(x_vals, line, label='Fit')
plt.ylabel('Fitness relative to WT', fontsize=8)
plt.xlabel(r'Generations $M$', fontsize=8)
plt.title(f'ParD3 fitness decay, $\\rho$ = {np.around(decay_rate[0]/2,2)}',fontsize=10)
plt.tick_params(axis='both', which='major', labelsize=6)
plt.legend(fontsize=8)
plt.ylim(0,1.1)
plt.show()

out_path = os.path.join(figures_dir, 'ParD3_decay.pdf')
plt.savefig(out_path, dpi=dpi)
print(f'Saved -> {out_path}')


# --- Cell 76: ParD3 strategy space ---
import matplotlib.pyplot as plt

plt.figure(figsize=(1.5, 1.5), dpi=300)
mean_sweep = sweep.mean(axis=(0,3))
plt.imshow(mean_sweep)

# --- Add scatter plot with square markers proportional to frequency ---
max_size = 100  # Adjust as needed
dot_sizes = (scipy_freq_matrix / scipy_freq_matrix.max()) * max_size

for i in range(5):  # rows
    for j in range(5):  # columns
        if scipy_freq_matrix[i, j] > 0:
            plt.scatter(
                j - 0.2, i,                 # offset left
                s=dot_sizes[i, j] * 0.2,
                color=c2,
                alpha=1,
                marker='o',
                label='SLIDE'  # avoid duplicate labels
            )

# Baseline marker (offset right)
plt.scatter(
    0 + 0.2, 4, s=20, color=c1, alpha=1, marker='o', label='Baseline'
)

# --- Optimal strategy marker as red circle ---
max_idx = np.unravel_index(np.argmax(mean_sweep, axis=None), mean_sweep.shape)
optimal_i, optimal_j = max_idx
plt.scatter(
    optimal_j, optimal_i,
    s=30,              # slightly larger to stand out
    color='red',
    alpha=1,
    marker='o',
    linewidth=0.5,
    label='Optimal'
)

# --- Formatting ---
plt.xticks([0, 4], labels=[0.0, 0.19])
plt.yticks([0, 4], labels=[20, 1])
plt.ylabel('No. sub populations', fontsize=labelsize - 1)
plt.xlabel('Base chance', fontsize=labelsize - 1)
plt.title('Strategy space', fontsize=titlesize - 1)
plt.tick_params(axis='both', which='major', labelsize=5)

# --- Legend outside, smaller ---
plt.legend(
    fontsize=6,
    loc='center left',
    bbox_to_anchor=(1.07, 0.5)
)

out_path = os.path.join(figures_dir, 'ParD3_strategy_space.pdf')
plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
print(f'Saved -> {out_path}')
plt.show()
