"""
Figure 3D — Step 3: plot rho prediction accuracy over mutation rate.

Input
-----
  plot_data/mut_accuracy.pkl
    (mut_decay_rates, muts)
      mut_decay_rates : array (25, 500) — fitted rho per replicate
      muts            : array (25,) — mutation rates (linspace 0.01 to 2)

Output
------
  figures/accuracy_over_mut.pdf

Usage
-----
  python figures_src/figure_3D_accuracy_over_mut/3_plot.py

Originally from ruggedness_figures_plots.ipynb cell 16.
Note: conv_p = 16/25 is carried over from cell 13 (popsize figure) — it is the
true (K+1)/N value for the NK landscape used in both the popsize and mut sweeps.
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

## Fontsizes
titlesize = 10
labelsize = 8
ticksize = 6
legendsize = 8
dpi = 350
plt.rcParams["font.family"] = "DejaVu Sans"

with open(os.path.join(plot_data_dir, 'mut_accuracy.pkl'), 'rb') as f:
    mut_decay_rates, muts = pickle.load(f)

y_means = np.mean(mut_decay_rates,axis=1)
y_stds = mut_decay_rates.std(axis=1)

conv_p = 16/25

# Step 3: Create Fill-Between Plot
plt.figure(figsize=(3.5,1.2), dpi=300)
plt.plot(muts, y_means, 'o-', label=r"Mean estimated $\rho$")  # Line plot with markers
plt.axhline(y=conv_p, label=r'$(K+1)/N$', c='red', alpha=0.4,linestyle='--')
plt.fill_between(muts, y_means - y_stds, y_means + y_stds, alpha=0.3, label="±1 Std Dev")  # Shaded error band
# plt.plot(muts, np.ones_like(muts) * 12/25, ls = '--', label="True")
plt.grid(True)
plt.legend(fontsize = legendsize-2, loc='lower right')
plt.tick_params(axis='both', which='major', labelsize=ticksize)
# plt.ylim(0.0,1.0)
# plt.xlim(0,1.0)
plt.xlabel(r'Mutations per cell per generation $\theta$', fontsize=labelsize)
plt.ylabel(r"Estimated $\rho$", fontsize=labelsize)
plt.ylim(0.55,0.7)
plt.title('Accuracy over mutation rate', fontsize=titlesize)

out_path = os.path.join(figures_dir, 'accuracy_over_mut.pdf')
plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
print(f'Saved → {out_path}')
