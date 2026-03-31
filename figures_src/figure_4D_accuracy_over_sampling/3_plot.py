"""
Figure 4D — Step 3: rho accuracy as a function of number of trajectories averaged.

Input
-----
  plot_data/trajectory_subsampling.pkl
    ld_results: list of 4 landscape results (GB1, TrpB, TEV, ParD3).
    Each entry is a list of 11 arrays of bootstrapped rho values across
    log-spaced trajectory counts.

  plot_data/fourier_spectra_empirical.pkl
    tuple of 4 spectra (GB1, TrpB, TEV, ParD3) — used to compute weighted
    average frequency (fourier_vals) for the dotted reference lines.

Output
------
  figures/accuracy_over_sampling.pdf

Usage
-----
  python figures_src/figure_4D_accuracy_over_sampling/3_plot.py

Originally from ruggedness_figures_plots.ipynb cells 30–31 (fourier_vals
computation) and cell 34 (main plot). fourier_vals is recomputed here from
the pkl rather than depending on cell 31 having been run.
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
c2 = 'tab:blue'
c1 = 'tab:orange'
c3 = '#f55f74'
c4 = 'tab:green'

titlesize = 10
labelsize = 8
ticksize  = 6
legendsize = 8
dpi = 350
plt.rcParams["font.family"] = "DejaVu Sans"

# Recompute fourier_vals (originally computed in cell 31 as side-effect)
with open(os.path.join(plot_data_dir, 'fourier_spectra_empirical.pkl'), 'rb') as f:
    spectra = pickle.load(f)

A = 20
N = 4
d = N * (A - 1)
fourier_vals = []
for n, i in enumerate(spectra):
    i = i[1:]
    weighted_avg = np.sum(A * (np.arange(len(i)) + 1) * i) / np.sum(i) / d
    fourier_vals.append(weighted_avg)

with open(os.path.join(plot_data_dir, 'trajectory_subsampling.pkl'), 'rb') as f:
    ld_results = pickle.load(f)

# ---- Plotting code from notebook cell 34 (exact copy) ----
import numpy as np
import matplotlib.pyplot as plt
data_names = ['GB1', 'TrpB', 'TEV', 'ParD3']
colours = [c1,c2,c4,c3]
trajectories = np.round(np.logspace(0, np.log10(160000), 11)).astype(int)
plt.figure(figsize=(3.5,3), dpi=300)

for h in range(3):

    means = []
    errs = []

    for t in range(len(trajectories)):
        vals = ld_results[h][t]     # (n_boot,)
        means.append(np.mean(vals))
        errs.append(np.std(vals))   # or SEM: np.std(vals) / np.sqrt(len(vals))

    means = np.array(means)
    errs  = np.array(errs)

    plt.plot(
        trajectories,
        means,
        label=data_names[h],
        color = colours[h]
    )

    plt.fill_between(
        trajectories,
        means - errs,
        means + errs,
        alpha=0.25,
        color = colours[h],
        edgecolor=None
    )

    #plt.hlines(
    #    y=fourier_vals[h] xmin=0, xmax=1000, color=colours[h], linestyle='--', alpha=0.4)

    plt.hlines(
        y=fourier_vals[h], xmin=0, xmax=160000, color=colours[h], linestyle='dotted',alpha=0.4)


trajectories = np.round(np.logspace(0, np.log10(8000), 11)).astype(int)

means = []
errs = []

for t in range(len(trajectories)):
    vals = ld_results[-1][t]     # (n_boot,)
    means.append(np.mean(vals))
    errs.append(np.std(vals))   # or SEM: np.std(vals) / np.sqrt(len(vals))

means = np.array(means)
errs  = np.array(errs)

plt.plot(
    trajectories,
    means,
    label=data_names[-1],
    color = colours[-1]
)

plt.fill_between(
    trajectories,
    means - errs,
    means + errs,
    alpha=0.25,
    color = colours[-1],
    edgecolor=None
)

#plt.hlines(
#    y=fourier_vals[h] xmin=0, xmax=1000, color=colours[h], linestyle='--', alpha=0.4)

plt.hlines(
    y=fourier_vals[-1], xmin=0, xmax=8000, color=colours[-1], linestyle='dotted',alpha=0.4)

plt.xlabel('Number of trajectories averaged', fontsize=labelsize)
plt.ylabel(r'Decay rate $\rho$', fontsize=labelsize)
plt.title(r'$\rho$ accuracy with increasing samples', fontsize=titlesize)
plt.legend(fontsize = legendsize)
plt.ylim(0,1.8)
plt.tick_params(axis='both', which='major', labelsize=ticksize)
plt.xscale('log')
# ---- End of notebook cell 34 ----

out_path = os.path.join(figures_dir, 'accuracy_over_sampling.pdf')
plt.savefig(out_path, dpi=dpi)
print(f'Saved → {out_path}')
