"""
Figure 3B — Step 3: plot rho prediction accuracy over NK ruggedness.

Input
-----
  plot_data/ruggedness_accuracy.pkl
    (k_plus_one_over_ns, decay_rates)
      k_plus_one_over_ns : array (100,) — (K+1)/N for each NK pair
      decay_rates        : array (100, reps) — fitted rho per replicate

Output
------
  figures/accuracy_over_K.pdf

Usage
-----
  python figures_src/figure_3B_accuracy_over_K/3_plot.py

Originally from ruggedness_figures_plots.ipynb cell 10.
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

with open(os.path.join(plot_data_dir, 'ruggedness_accuracy.pkl'), 'rb') as f:
    k_plus_one_over_ns, decay_rates = pickle.load(f)

# Step 1: Get ordering for rho
arg_sort_kn = np.argsort(k_plus_one_over_ns)
sorted_rho = k_plus_one_over_ns[arg_sort_kn]
sorted_decay = decay_rates[arg_sort_kn]

# Step 2: Group the values
grouped_rho = sorted_rho.reshape(10,-1)
grouped_decay = sorted_decay.reshape(10,-1)

mean_rho = np.mean(grouped_rho, axis = 1)
mean_decay = np.mean(grouped_decay, axis = 1)
std_decay = np.std(grouped_decay, axis = 1)

true_k_over_n = np.linspace(0.1,1,10)

# Step 3: Create Fill-Between Plot
plt.figure(figsize=(3.5,3), dpi=300)
plt.plot(true_k_over_n, mean_decay, 'o-', label=r"Mean estimated $\rho$")  # Line plot with markers
plt.fill_between(true_k_over_n, mean_decay - std_decay, mean_decay + std_decay, alpha=0.3, label="±1 Std dev")  # Shaded error band

plt.plot(true_k_over_n, mean_rho, c='red', alpha=0.4,linestyle='--', label=r'$(K+1)/N$')

# Labels and Title
plt.xlabel(r"$(K+1)/N$", fontsize=labelsize)
plt.ylabel(r"Estimated $\rho$", fontsize=labelsize)
plt.title(r"$\rho$ prediction accuracy over ruggedness", fontsize = titlesize)
plt.legend(fontsize=legendsize)
plt.tick_params(axis='both', which='major', labelsize=ticksize)
plt.grid(True)

out_path = os.path.join(figures_dir, 'accuracy_over_K.pdf')
plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
print(f'Saved → {out_path}')
