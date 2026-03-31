"""
Figure 3A — Step 3: plot example fitness decay curves for smooth vs rugged NK.

Input
-----
  plot_data/smooth_rugged_example.pkl
    (smooth_rugged, fitted_lines)
      smooth_rugged : array (2, 25) — normalised mean fitness per generation
      fitted_lines  : array (2, 25) — fitted exponential decay curves

Output
------
  figures/decay_curves_example.pdf

Usage
-----
  python figures_src/figure_3A_decay_curves_example/3_plot.py

Originally from ruggedness_figures_plots.ipynb cell 7.
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

## Plot colours
c2 = 'tab:blue'#298c8c'
c1 = 'tab:orange' #800074'
c3 = '#f55f74'
c4 = 'tab:green'

## Fontsizes
titlesize = 10
labelsize = 8
ticksize = 6
legendsize = 8
dpi = 350
plt.rcParams["font.family"] = "DejaVu Sans"

with open(os.path.join(plot_data_dir, 'smooth_rugged_example.pkl'), 'rb') as f:
    smooth_rugged, fitted_lines = pickle.load(f)

plt.figure(figsize=(3.5,3), dpi=300)

generations = np.linspace(1,25,25)

#plt.plot(generations, smooth_rugged[1], label = r"$K/N = 0.1$", c=c2)
plt.scatter(generations, smooth_rugged[1], label = r"$(K+1)/N = 0.1$", c=c2,s=5)
plt.plot(generations, fitted_lines[1], c=c2)
plt.scatter(generations, smooth_rugged[0], label = r"$(K+1)/N = 0.75$", c=c1,s=5)
plt.plot(generations, fitted_lines[0], c=c1)
plt.legend(fontsize = legendsize)

plt.title('Fitness decay curves', fontsize = titlesize)
plt.xlabel('Generations $M$', fontsize=labelsize)
plt.ylabel(r'Fitness $F_\mu$', fontsize=labelsize)
plt.tick_params(axis='both', which='major', labelsize=ticksize)

out_path = os.path.join(figures_dir, 'decay_curves_example.pdf')
plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
print(f'Saved → {out_path}')
