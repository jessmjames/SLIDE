"""
Figure 4F — Step 3: rho variance over generations sampled.

Input
-----
  plot_data/estimation_variance.pkl
    (array_results1, array_results2): array_results2 used here.
    array_results2 is a list of 4 arrays (one per landscape: GB1, TrpB, TEV, ParD3),
    each of shape (n_generations, n_replicates).

  plot_data/fourier_spectra_empirical.pkl
    tuple of 4 spectra (GB1, TrpB, TEV, ParD3) — used to compute weighted
    average frequency (fourier_vals) for the dotted reference lines.

Output
------
  figures/empirical_variance_over_generations.pdf

Usage
-----
  python figures_src/figure_4F_empirical_variance_generations/3_plot.py

Originally from ruggedness_figures_plots.ipynb cells 2, 3, 30, 31 (partial),
39, 41.
"""

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

import pickle
import numpy as np
import matplotlib.pyplot as plt

figures_dir = os.path.join(parent_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

c2 = 'tab:blue'
c1 = 'tab:orange'
c3 = '#f55f74'
c4 = 'tab:green'

titlesize = 10
labelsize = 8
ticksize = 6
legendsize = 8
dpi = 350
plt.rcParams["font.family"] = "DejaVu Sans"

with open(os.path.join(parent_dir, 'plot_data/fourier_spectra_empirical.pkl'), 'rb') as f:
    spectra = pickle.load(f)

colours = [c1,c2,c4,c3]
labels = ['GB1', 'TrpB', 'TEV','ParD3']
markers = ['o','s','^','D']

A=20
N=[4,4,4,3]
fourier_vals = []

for n,i in enumerate(spectra):
    d = N[n]*(A-1)
    i = i[1:]
    i_norm = (i - i.min()) / (i.max() - i.min())
    indexes = np.arange(len(i))+1
    weighted_avg = np.sum(A * indexes * i) / np.sum(i) / d
    fourier_vals.append(weighted_avg)

with open(os.path.join(parent_dir, 'plot_data/estimation_variance.pkl'), 'rb') as f:
    (array_results1, array_results2) = pickle.load(f)

plt.figure(figsize=(3.5, 3), dpi=dpi)

generations = np.arange(1, 24)

for n, out1 in enumerate(array_results2):
    means = out1.mean(axis=1)
    stds = out1.std(axis=1)

    plt.plot(generations, means, label=labels[n], marker=markers[n], markersize=3, color=colours[n])
    plt.fill_between(generations, means - stds, means + stds, alpha=0.3, color=colours[n], edgecolor=None)
    plt.hlines(y=fourier_vals[n], xmin=0, xmax=23, color=colours[n], linestyle='dotted', alpha=0.4)

plt.xlabel("Generations sampled", fontsize=labelsize)
plt.ylabel(r"Decay rate $\rho$", fontsize=labelsize)
plt.title("Variance over generations sampled", fontsize=titlesize)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
plt.legend(fontsize=legendsize)
plt.ylim(0,1.1)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'empirical_variance_over_generations.pdf'), dpi=dpi)
