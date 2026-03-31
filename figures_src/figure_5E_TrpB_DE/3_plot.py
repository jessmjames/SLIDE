"""
Figure 5E — Step 3: plot TrpB directed evolution (decay curve, strategy space, DE).

Input
-----
  plot_data/TrpB_strategy_selection.pkl
    (x_vals, decay_mean, decay_rate, sweep, scipy_freq_matrix, run, scatter, line,
     TrpB_decay_multi)

Output
------
  figures/TrpB_decay.pdf
  figures/TrpB_strategy_space.pdf
  figures/TrpB_DE.pdf

Usage
-----
  python figures_src/figure_5E_TrpB_DE/3_plot.py

Originally from ruggedness_figures_plots.ipynb cells 63, 64, 65.
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

with open(os.path.join(plot_data_dir, 'TrpB_strategy_selection.pkl'), 'rb') as f:
    x_vals, decay_mean, decay_rate, sweep, scipy_freq_matrix, run, scatter, line, TrpB_decay_multi = pickle.load(f)

c1 = 'tab:orange'
c2 = 'tab:blue'
titlesize = 10
labelsize = 8
ticksize = 6
legendsize = 8
dpi = 350

# --- Cell 63: TrpB decay curve ---
plt.figure(figsize=(3, 1.5), dpi=300)
plt.scatter(x_vals, scatter, label='Mean fitness', s=15)
plt.plot(x_vals, line, label='Fit')
plt.ylabel('Fitness relative to WT', fontsize=8)
plt.xlabel(r'Generations $M$', fontsize=8)
plt.title(f'TrpB fitness decay, $\\rho$ = {np.around(decay_rate[0]/2,2)}',fontsize=10)
plt.tick_params(axis='both', which='major', labelsize=6)
plt.legend(fontsize=8)
plt.ylim(0,1.1)
plt.show()

out_path = os.path.join(figures_dir, 'TrpB_decay.pdf')
plt.savefig(out_path, dpi=dpi)
print(f'Saved -> {out_path}')


# --- Cell 64: TrpB strategy space ---
plt.figure(figsize=(1.5, 1.5), dpi=300)
mean_sweep = sweep.mean(axis=(0,3))
plt.imshow(mean_sweep)

# Add scatter plot with square markers proportional to frequency
max_size = 100  # Adjust as needed
dot_sizes = (scipy_freq_matrix / scipy_freq_matrix.max()) * max_size

for i in range(7):  # Loop over rows
    for j in range(7):  # Loop over columns
        if scipy_freq_matrix[i, j] > 0:  # Only plot if frequency > 0
            plt.scatter(
                j, i,
                s=dot_sizes[i, j]*0.2,
                color=c2,
                alpha=1,
                marker='o',
                label='SLIDE'
            )

# Baseline as a square
plt.scatter(0,6, s=20, color=c1, alpha=1, marker='o', label='Baseline')

# ---- Optimal strategy as a red square ----
max_idx = np.unravel_index(np.argmax(mean_sweep, axis=None), mean_sweep.shape)
optimal_i, optimal_j = max_idx
plt.scatter(
    optimal_j, optimal_i,
    s=30,              # slightly larger to stand out
    color='red',
    alpha=1,
    marker='o',
    linewidth=0.5,
    label='Optimum'
)

# Formatting
plt.xticks([0, 6], labels=[0.0, 0.19])
plt.yticks([0, 6], labels=[24, 1])
plt.ylabel('No. sub populations', fontsize=labelsize - 1)
plt.xlabel('Base chance', fontsize=labelsize - 1)
plt.title('Relative strategy performance', fontsize=titlesize - 2)
plt.tick_params(axis='both', which='major', labelsize=5)

# Legend outside, smaller
plt.legend(
    fontsize=6,
    loc='center left',
    bbox_to_anchor=(1.07, 0.5)
)

out_path = os.path.join(figures_dir, 'TrpB_strategy_space.pdf')
plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
print(f'Saved -> {out_path}')
plt.show()


# --- Cell 65: TrpB directed evolution ---
# --- Compute mean over reps first ---
run_startmean = [
    [
        np.mean(strategy, axis=0)  # mean over reps for each strategy
        for strategy in start
    ]
    for start in run
]

n_strat = len(run_startmean[0])
n_steps = max(len(start[0]) for start in run_startmean)  # max number of steps across starts

strategy_mean = []
strategy_std  = []

# --- Compute mean/std across starts ---
for s in range(n_strat):
    # Gather the mean-over-reps trajectories for this strategy
    start_vals = [start[s] for start in run_startmean]  # list of arrays (steps)

    # Pad sequences to the same length with NaN (optional)
    max_len = max(len(traj) for traj in start_vals)
    start_vals_pad = [list(traj) + [np.nan]*(max_len - len(traj)) for traj in start_vals]

    # Compute mean and std at each step, ignoring NaNs
    mean_vals = [np.nanmean([traj[step] for traj in start_vals_pad]) for step in range(max_len)]
    std_vals  = [np.nanstd([traj[step] for traj in start_vals_pad], ddof=1) for step in range(max_len)]

    strategy_mean.append(mean_vals)
    strategy_std.append(std_vals)

# --- Plotting ---
plt.figure(figsize=(4, 1.5), dpi=300)
steps = np.arange(n_steps)
colors = [c1, c2]
labels = ['Baseline', 'SLIDE']

for s in range(n_strat):
    plt.plot(steps, strategy_mean[s], label=labels[s], c=colors[s])
    plt.fill_between(
        steps,
        [m - std for m, std in zip(strategy_mean[s], strategy_std[s])],
        [m + std for m, std in zip(strategy_mean[s], strategy_std[s])],
        alpha=0.3,
        color=colors[s],
        linewidth=0
    )

plt.ylabel('Fitness relative to WT', fontsize=labelsize)
plt.xlabel(r'Generations $M$', fontsize=labelsize)
plt.title('TrpB directed evolution, 10 start average', fontsize=titlesize)
plt.tick_params(axis='both', which='major', labelsize=ticksize)
plt.legend(fontsize=8)

out_path = os.path.join(figures_dir, 'TrpB_DE.pdf')
plt.savefig(out_path, dpi=dpi)
print(f'Saved -> {out_path}')
plt.show()
