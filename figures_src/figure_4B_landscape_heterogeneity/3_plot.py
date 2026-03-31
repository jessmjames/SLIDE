"""
Figure 4B — Step 3: violin plot of rho distributions across NK and empirical landscapes.

Input
-----
  plot_data/heterogeneity_data.pkl  — (NK_rhos, empirical_rhos)
    NK_rhos: list of 4 rho distributions for NK landscapes at K/N = 0.25,
             0.50, 0.75, 1.00
    empirical_rhos: list of 4 rho distributions for GB1, TrpB, TEV, ParD3

Output
------
  figures/landscape_heterogeneity.pdf

Usage
-----
  python figures_src/figure_4B_landscape_heterogeneity/3_plot.py

Canonical notebook cell: ruggedness_figures_plots.ipynb cell 40
(the final savefig call among cells 38–40 that all write to the same file).
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

# Style settings (from notebook cells 2–3)
titlesize = 10
labelsize = 8
ticksize  = 6
legendsize = 8
dpi = 350
plt.rcParams["font.family"] = "DejaVu Sans"

with open(os.path.join(plot_data_dir, 'heterogeneity_data.pkl'), 'rb') as f:
    NK_rhos, empirical_rhos = pickle.load(f)

empirical_rhos = [np.clip(i, 0, 1) for i in empirical_rhos]
NK_rhos        = [np.clip(i, 0, 1) for i in NK_rhos]

# ---- Plotting code from notebook cell 40 (exact copy) ----
plt.figure(figsize=(3.5, 3), dpi=300)

data = NK_rhos + empirical_rhos
var_data = [np.std(i) for i in data]

plt.violinplot(data, showmeans=True)

plt.xticks([1, 2, 3,4,5,6,7,8], ['0.25', '0.50', '0.75','1.00','GB1','TrpB','TEV', 'PardD3'], fontsize=ticksize)
plt.ylabel(r'$\rho$ Estimation', fontsize = labelsize)
plt.title('Landscape heterogeneity', fontsize=titlesize)

for i, var in enumerate(var_data):
    x = i + 1  # violin positions start at 1
    y = max(data[i]) * 1.05  # slightly above the top
    plt.text(x, y, f'$\sigma$={var:.2f}', ha='center', va='bottom', fontsize=4.5)

plt.axvline(x=4.5, color='gray', linestyle='--', linewidth=1)
plt.tick_params(axis='both', which='major', labelsize=ticksize)
plt.ylim(-0.1,1.15)
plt.tight_layout()
# ---- End of notebook cell 40 ----

out_path = os.path.join(figures_dir, 'landscape_heterogeneity.pdf')
plt.savefig(out_path, dpi=dpi)
print(f'Saved → {out_path}')
