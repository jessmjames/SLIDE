"""
Figure 5C — Step 3: plot directed evolution on rugged NK landscape (N=45, K=25).

Input
-----
  plot_data/NK_DE.pkl
    list of 4 arrays: [K1_baseline, K25_baseline, K1_SLIDE, K25_SLIDE]

  plot_data/NK_strategy_spaces.pkl
    (smooth_strategies, rugged_strategies)

Output
------
  figures/N45K25_DE_fitness.pdf
  figures/N45K25_DE_strategy_space.pdf

Usage
-----
  python figures_src/figure_5C_NK_rugged_DE/3_plot.py

Originally from ruggedness_figures_plots.ipynb cells 52, 53.
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

with open(os.path.join(plot_data_dir, 'NK_DE.pkl'), 'rb') as f:
    DE_data = pickle.load(f)

with open(os.path.join(plot_data_dir, 'NK_strategy_spaces.pkl'), 'rb') as f:
    smooth_strategies, rugged_strategies = pickle.load(f)

c1 = 'tab:orange'
c2 = 'tab:blue'
dpi = 350

# --- Cell 52: N45K25 DE fitness ---
plt.figure(figsize=(3,3), dpi=300)
plt.plot(DE_data[1], label = 'Baseline strategy', color=c1)
plt.plot(DE_data[3], label = 'SLIDE', color=c2)
plt.ylabel('Arbitrary fitness scale', fontsize=8)
plt.xlabel(r'Generations $M$', fontsize=8)
plt.title('N = 45, K = 25',fontsize=10)
plt.tick_params(axis='both', which='major', labelsize=6)
plt.legend(fontsize=8)

out_path = os.path.join(figures_dir, 'N45K25_DE_fitness.pdf')
plt.savefig(out_path, dpi=dpi)
print(f'Saved -> {out_path}')

# --- Cell 53: N45K25 strategy space ---
plt.figure(figsize = (1.5,1.5), dpi=300)
plt.imshow(rugged_strategies.mean(axis=1).reshape(7,7))
plt.xticks([0,6], labels = [0.0,0.19])
plt.yticks([0,6], labels = [24,1])
plt.ylabel('No. sub populations', fontsize=8)
plt.xlabel('Base chance', fontsize=8)
plt.title('Strategy space')
plt.tick_params(axis='both', which='major', labelsize=6)

out_path = os.path.join(figures_dir, 'N45K25_DE_strategy_space.pdf')
plt.savefig(out_path, dpi=dpi)
print(f'Saved -> {out_path}')
