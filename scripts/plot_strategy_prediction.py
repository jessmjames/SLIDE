"""
Pipeline: large_strategy_sweep_100.pkl + large_decay_curve_sweep.pkl
         -> plot_data/strategy_prediction_accuracy.pkl
         -> figures/strategy_prediction.pdf

Run after:
  1. scripts/large_strategy_sweep.py      (-> large_strategy_sweep_100.pkl)
  2. scripts/large_decay_curve_sweep.py   (-> large_decay_curve_sweep.pkl)
"""

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import pickle
import numpy as np
import matplotlib.pyplot as plt
from direvo_functions import get_single_decay_rate, base_chance_threshold_fixed_prop

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
slide_data_dir = "./"
plot_data_dir  = os.path.join(parent_dir, 'plot_data')
figures_dir    = os.path.join(parent_dir, 'figures')

# ---------------------------------------------------------------------------
# NK grid (must match large_strategy_sweep.py and large_decay_curve_sweep.py)
# ---------------------------------------------------------------------------
import jax.numpy as jnp

def NK_grid(N_range, num_samples=10):
    N = jnp.linspace(N_range[0], N_range[1], num=num_samples)
    K = jnp.array([jnp.linspace(1, i, num_samples) for i in N]).reshape(num_samples, num_samples)
    N = jnp.repeat(N, num_samples).reshape(num_samples, num_samples)
    return N, K

N_grid, K_grid = NK_grid([10, 50])
Ns, Ks = N_grid.flatten(), K_grid.flatten()
Ns = jnp.flip(Ns)
Ks = jnp.flip(Ks)
NKs = np.array(list(zip(Ns, Ks)))           # (100, 2)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading strategy sweep...")
with open(os.path.join(slide_data_dir, "large_strategy_sweep_100.pkl"), "rb") as f:
    strategy_data = pickle.load(f)   # (100, 7, 7, 300)

print("Loading decay curve sweep...")
with open(os.path.join(slide_data_dir, "large_decay_curve_sweep.pkl"), "rb") as f:
    decay_data = pickle.load(f)      # (100, 25, 10, 25)

# ---------------------------------------------------------------------------
# Pre-process decay curves
# ---------------------------------------------------------------------------
reshaped_decay   = decay_data.reshape(100, -1, 25)                   # (100, 250, 25)
normalized_decay = reshaped_decay / reshaped_decay[:, :, 0:1]        # normalise t=0 -> 1

# ---------------------------------------------------------------------------
# N-average to get 10 K-bins
# ---------------------------------------------------------------------------
reshaped_strategies = strategy_data.reshape(100, -1, 300)            # (100, 49, 300)
# Average over the 9 N values for each K-bin (first 90 NK combos cover 9N × 10K)
N_meaned_strategies = reshaped_strategies[:90].mean(axis=2).reshape(9, 10, 49).mean(axis=0)  # (10, 49)
N_meaned_decay      = normalized_decay.reshape(10, 10, 250, 25).mean(axis=(0, 2))            # (10, 25)

# ---------------------------------------------------------------------------
# Fit decay rates for each K-bin -> lookup table
# ---------------------------------------------------------------------------
thresholds, base_chances = base_chance_threshold_fixed_prop([0, 0.19], 0.2, 7)
splits = [24, 20, 16, 12, 8, 4, 1]

decay_rates = np.array([get_single_decay_rate(curve, mut=0.5)[0] for curve in N_meaned_decay])

base_chance_array = np.tile(base_chances, 7)
splitting_array   = np.repeat(splits, 7)

optimal_splits        = []
optimal_base_chances  = []
for i in range(10):
    best = N_meaned_strategies[i].argmax()
    optimal_splits.append(splitting_array[best])
    optimal_base_chances.append(base_chance_array[best])

optimal_splits       = np.array(optimal_splits)
optimal_base_chances = np.array(optimal_base_chances)

with open(os.path.join(plot_data_dir, 'optimal_DE_strategies.pkl'), 'wb') as f:
    pickle.dump((decay_rates, optimal_splits, optimal_base_chances), f)
print("Saved plot_data/optimal_DE_strategies.pkl")

# ---------------------------------------------------------------------------
# Predict strategy for each NK landscape × run
# ---------------------------------------------------------------------------
k_over_ns = (NKs[:, 1] + 1) / NKs[:, 0]   # true rho proxy

predicted_base_chances = []
predicted_splittings   = []
actual_k_over_ns       = []

for landscape in range(normalized_decay.shape[0]):
    for run in range(100):
        run_data  = normalized_decay[landscape, run, :]
        run_rho   = get_single_decay_rate(run_data, mut=0.5)[0]
        best_idx  = np.argmin(np.abs(decay_rates - run_rho))
        predicted_base_chances.append(optimal_base_chances[best_idx])
        predicted_splittings.append(optimal_splits[best_idx])
        actual_k_over_ns.append(k_over_ns[landscape])

predicted_base_chances = np.array(predicted_base_chances)
predicted_splittings   = np.array(predicted_splittings)
actual_k_over_ns       = np.array(actual_k_over_ns)

# Bin by rounded k/n
rounded = np.round(actual_k_over_ns, 1)
unique_x = np.unique(rounded)

bc_means = [predicted_base_chances[rounded == x].mean() for x in unique_x]
bc_stds  = [predicted_base_chances[rounded == x].std()  for x in unique_x]
sp_means = [predicted_splittings[rounded == x].mean()   for x in unique_x]
sp_stds  = [predicted_splittings[rounded == x].std()    for x in unique_x]

with open(os.path.join(plot_data_dir, 'strategy_prediction_accuracy.pkl'), 'wb') as f:
    pickle.dump((actual_k_over_ns, bc_means, bc_stds, sp_means, sp_stds), f)
print("Saved plot_data/strategy_prediction_accuracy.pkl")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
c1 = 'tab:orange'
c2 = 'tab:blue'
titlesize = 10
labelsize = 8
ticksize  = 6
legendsize = 8
dpi = 350

fig, ax1 = plt.subplots(figsize=(3.5, 3), dpi=dpi)

ax1.set_xlabel(r'True $\rho$', fontsize=labelsize)
ax1.set_ylabel(r'Predicted base chance $b$', color=c1, fontsize=labelsize)
ax1.errorbar(unique_x, bc_means, yerr=bc_stds, fmt='o', capsize=5, color=c1)
ax1.plot(decay_rates, optimal_base_chances, color=c1, linestyle='--', alpha=0.6,
         label='Optimal base chance')
ax1.tick_params(axis='y', labelcolor=c1, labelsize=ticksize)
ax1.tick_params(axis='x', labelsize=ticksize)
ax1.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.3)

ax2 = ax1.twinx()
ax2.set_ylabel('Predicted splitting (total 1200)', color=c2, fontsize=labelsize)
ax2.errorbar(unique_x, sp_means, yerr=sp_stds, fmt='o', capsize=5, color=c2)
ax2.plot(decay_rates, optimal_splits, color=c2, linestyle='--', alpha=0.6,
         label='Optimal splitting')
ax2.tick_params(axis='y', labelcolor=c2, labelsize=ticksize)

ax1.set_title(r'Predicted DE strategies vs true $\rho$', fontsize=titlesize)
ax1.legend(fontsize=legendsize - 2, loc='upper left', bbox_to_anchor=(0.0, 1.0))
ax2.legend(fontsize=legendsize - 2, loc='upper left', bbox_to_anchor=(0.0, 0.9))
plt.xlim(0.05, 1.05)
fig.tight_layout()

out_path = os.path.join(figures_dir, 'strategy_prediction.pdf')
plt.savefig(out_path, dpi=dpi)
print(f"Saved {out_path}")
