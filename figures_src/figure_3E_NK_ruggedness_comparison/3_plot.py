"""
Figure 3E — Step 3: plot normalised NK ruggedness metric comparison.

Input
-----
  plot_data/NK_ruggedness_metric_comparison.pkl
    (NK_roughness_to_slope, NK_fourier, convergence_rates, NK_paths_to_max,
     NK_closest_max, k_over_ns, NK_le_normed)
    All arrays shape (12,) — one value per K in [0, 11] for N=12, all
    min-max normalised to [0, 1].

Output
------
  figures/NK_ruggedness_metric_comparison.pdf

Usage
-----
  python figures_src/figure_3E_NK_ruggedness_comparison/3_plot.py

Originally from ruggedness_figures_plots.ipynb cell 19.
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

with open(os.path.join(plot_data_dir, 'NK_ruggedness_metric_comparison.pkl'), 'rb') as f:
    NK_roughness_to_slope, NK_fourier, convergence_rates, NK_paths_to_max, NK_closest_max, k_over_ns, NK_le_normed = pickle.load(f)

plt.figure(figsize=(3.5,3), dpi=300)
plt.plot(k_over_ns, convergence_rates/convergence_rates.max(), label = r'$\rho$ (Decay rate)')
plt.plot(k_over_ns, 1 - NK_fourier/NK_fourier.max(), label = r'1 - Landscape ${R}^{2}$', alpha=0.6, linestyle='--')
plt.plot(k_over_ns, NK_roughness_to_slope/NK_roughness_to_slope.max(), label = 'Roughness to slope ratio', alpha=0.6, linestyle='--')
plt.plot(k_over_ns, 1-NK_paths_to_max/NK_paths_to_max.max(), label='1 - Paths to max', alpha=0.6, linestyle='--')
plt.plot(k_over_ns, NK_closest_max/NK_closest_max.max(), label='1 - Dist. to closest local max', alpha=0.6, linestyle='--')
plt.plot(k_over_ns, NK_le_normed/NK_paths_to_max.max(), label='Local Epistasis', alpha=0.6, linestyle='--')
plt.legend(loc = 'lower right', fontsize=legendsize-2)
plt.title('Ruggedness metric comparison', fontsize=titlesize)
plt.xlabel(r'$(K+1)/N$', fontsize = labelsize)
plt.ylabel('Normalised ruggedness measurements', fontsize = labelsize)
plt.tick_params(axis='both', which='major', labelsize=ticksize)

out_path = os.path.join(figures_dir, 'NK_ruggedness_metric_comparison.pdf')
plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
print(f'Saved → {out_path}')
