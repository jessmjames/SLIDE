"""
Figure 4C — Step 3: violin plot of rho estimation across landscapes.

Input
-----
  plot_data/heterogeneity_data.pkl  — (NK_rhos, empirical_rhos)

Output
------
  figures/repeat_heterogeneity.pdf

Usage
-----
  python figures_src/figure_4C_repeat_heterogeneity/3_plot.py

Originally from ruggedness_figures_plots_IK.ipynb cell 44.
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

titlesize = 10
labelsize = 8
ticksize  = 6

with open(os.path.join(plot_data_dir, 'heterogeneity_data.pkl'), 'rb') as f:
    NK_rhos, empirical_rhos = pickle.load(f)

data     = NK_rhos + empirical_rhos
var_data = [np.std(d) for d in data]

fig = plt.figure(figsize=(3.5, 3), dpi=300)

vp = plt.violinplot(data, showmeans=True, showextrema=True)
vp['cbars'].set_linewidth(0.5)
vp['cmins'].set_linewidth(0.5)
vp['cmaxes'].set_linewidth(0.5)
vp['cmeans'].set_linewidth(0.5)

plt.xticks(
    [1, 2, 3, 4, 5, 6, 7, 8],
    ['0.25', '0.50', '0.75', '1.00', 'GB1', 'TrpB', 'TEV', 'ParD3'],
    fontsize=ticksize,
)
plt.ylabel(r'$\rho$ Estimation', fontsize=labelsize)
plt.title('Landscape heterogeneity', fontsize=titlesize)

for i, var in enumerate(var_data):
    x = i + 1
    y = max(data[i]) * 1.05
    plt.text(x, y, f'$\\sigma$={var:.2f}', ha='center', va='bottom', fontsize=4.5)

plt.axvline(x=4.5, color='gray', linestyle='--', linewidth=0.5)
plt.tick_params(axis='both', which='major', labelsize=ticksize)
plt.ylim(-0.1, 1.15)
plt.tight_layout()

out = os.path.join(figures_dir, 'repeat_heterogeneity.pdf')
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f'Saved → {out}')
