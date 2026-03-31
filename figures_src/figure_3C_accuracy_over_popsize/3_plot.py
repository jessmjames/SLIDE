"""
Figure 3C — Step 3: plot rho prediction accuracy over population size.

Input
-----
  plot_data/popsize_accuracy.pkl
    (popsize_decay_rates, pops)
      popsize_decay_rates : array (25, 500) — fitted rho per replicate
      pops                : array (25,) — population sizes

Output
------
  figures/accuracy_over_popsize.pdf

Usage
-----
  python figures_src/figure_3C_accuracy_over_popsize/3_plot.py

Originally from ruggedness_figures_plots.ipynb cell 13.
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

with open(os.path.join(plot_data_dir, 'popsize_accuracy.pkl'), 'rb') as f:
    popsize_decay_rates, pops = pickle.load(f)

y_means = popsize_decay_rates.mean(axis=1)
y_stds = popsize_decay_rates.std(axis=1)
pop_y_means = y_means
conv_p = 16/25
# Step 3: Create Fill-Between Plot
plt.figure(figsize=(3.5,1.2), dpi=300)
plt.plot(pops, y_means, 'o-', label=r"Mean estimated $\rho$")  # Line plot with markers
plt.axhline(y=conv_p, label=r'$(K+1)/N$', c='red', alpha=0.4, linestyle='--')
# plt.plot(pops, np.ones_like(pops) * 12/25, ls = '--', label="True")
plt.fill_between(pops, y_means - y_stds, y_means + y_stds, alpha=0.3, label="±1 Std Dev")  # Shaded error band
plt.grid(True)
plt.legend(fontsize = legendsize-2, loc="upper right")
plt.tick_params(axis='both', which='major', labelsize=ticksize)

plt.title('Accuracy over population size', fontsize = titlesize)
plt.ylabel(r"Estimated $\rho$", fontsize=labelsize)
plt.xlabel('Population size', fontsize=labelsize)

out_path = os.path.join(figures_dir, 'accuracy_over_popsize.pdf')
plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
print(f'Saved → {out_path}')
