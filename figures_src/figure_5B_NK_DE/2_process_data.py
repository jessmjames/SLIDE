"""
Figure 5B — Step 2: derive NK DE strategies and run directed evolution (smooth, K=1).

Runs the full NK strategy-prediction and directed-evolution pipeline:
  1. Load large_strategy_sweep.pkl + large_decay_curve_sweep.pkl.
  2. Compute N-averaged optimal DE strategies (base chance + split) from the sweep.
  3. Predict per-landscape strategies by estimating rho from decay curves.
  4. Run DE for 4 conditions: (N=45, K=1) and (N=45, K=25), each with a
     baseline strategy (bc=0, split=1) and the SLIDE-predicted strategy.
  5. Extract the winning split from each run and compute mean fitness trajectory.

Outputs both NK_DE.pkl (fitness trajectories) and NK_strategy_spaces.pkl
(raw strategy-space matrices for the smooth and rugged landscapes).

Fig 5C uses the same output files — run this script once for both.

Input
-----
  SLIDE_data/large_strategy_sweep.pkl
  SLIDE_data/large_decay_curve_sweep.pkl

Output
------
  plot_data/NK_DE.pkl
    list of 4 arrays [K1_baseline, K25_baseline, K1_SLIDE, K25_SLIDE]
    each shape (50,) — mean fitness over steps (winning split selected post-hoc)

  plot_data/NK_strategy_spaces.pkl
    (smooth_strategies, rugged_strategies)
    smooth_strategies: reshaped_strategies[19], rugged_strategies: reshaped_strategies[14]

Source
------
  ruggedness_figures_data_processing.ipynb, cells 5–6, 11, 80–82, 94, 98–104.
"""

import os
import sys
import pickle
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

from direvo_functions import (
    build_NK_landscape_function,
    build_mutation_function,
    build_selection_function,
    run_directed_evolution,
    base_chance_threshold_fixed_prop,
)
import selection_function_library as slct
from slide_config import get_slide_data_dir

slide_data_dir = get_slide_data_dir()
plot_data_dir  = os.path.join(parent_dir, 'plot_data')
os.makedirs(plot_data_dir, exist_ok=True)

out_de     = os.path.join(plot_data_dir, 'NK_DE.pkl')
out_strat  = os.path.join(plot_data_dir, 'NK_strategy_spaces.pkl')

if os.path.exists(out_de) and os.path.exists(out_strat):
    print(f'Already exists: {out_de}')
    print(f'Already exists: {out_strat}')
    sys.exit(0)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def NK_grid(N_range, num_samples=10):
    N = jnp.linspace(N_range[0], N_range[1], num=num_samples)
    K = jnp.array([jnp.linspace(1, i, num_samples) for i in N]).reshape(num_samples, num_samples)
    N = jnp.repeat(N, num_samples).reshape(num_samples, num_samples)
    return N, K


# Modified directedEvolution that supports splits (notebook cell 99).
# rng_rep seeds the landscape; s_rng seeds the evolutionary run.
def directedEvolution(s_rng,
                      rng_rep,
                      selection_strategy,
                      selection_params,
                      empirical=False,
                      N=None,
                      K=None,
                      landscape=None,
                      popsize=100,
                      mut_chance=0.01,
                      num_steps=50,
                      num_reps=10,
                      define_i_pop=None,
                      average=True):

    if define_i_pop is None:
        i_pop = jnp.array([jr.randint(rng_rep, (N,), 0, 2)] * popsize)
    else:
        i_pop = define_i_pop

    if empirical:
        from direvo_functions import build_empirical_landscape_function
        fitness_function = build_empirical_landscape_function(landscape)
        mutation_function = build_mutation_function(mut_chance, 20)
    else:
        fitness_function = build_NK_landscape_function(rng_rep, N, K)
        mutation_function = build_mutation_function(mut_chance, 2)

    selection_function = build_selection_function(selection_strategy, selection_params)

    vmapped_run = jax.jit(jax.vmap(lambda r: run_directed_evolution(
        r, i_pop, selection_function, mutation_function,
        fitness_function=fitness_function, num_steps=num_steps)[1]))

    rng_seeds = jr.split(s_rng, num_reps)
    results = vmapped_run(rng_seeds)
    return results


# ---------------------------------------------------------------------------
# Step 1: load data
# ---------------------------------------------------------------------------

print('Loading large_strategy_sweep.pkl ...')
with open(os.path.join(slide_data_dir, 'large_strategy_sweep.pkl'), 'rb') as f:
    strategy_data = pickle.load(f)

print('Loading large_decay_curve_sweep.pkl ...')
with open(os.path.join(slide_data_dir, 'large_decay_curve_sweep.pkl'), 'rb') as f:
    decay_data = pickle.load(f)

# ---------------------------------------------------------------------------
# Step 2: build NK grid and normalise decay curves (cells 5–6, 11)
# ---------------------------------------------------------------------------

N_grid, K_grid = NK_grid([10, 50])
Ns = jnp.flip(N_grid.flatten())
Ks = jnp.flip(K_grid.flatten())
NKs = list(zip(Ns, Ks))

reshaped_decay_curves = decay_data.reshape(100, -1, 25)
normalized_decay_curves = reshaped_decay_curves / reshaped_decay_curves[:, :, 0][:, :, np.newaxis]

# ---------------------------------------------------------------------------
# Step 3: N-averaged optimal strategies (cell 80–82)
# ---------------------------------------------------------------------------

from direvo_functions import get_single_decay_rate

reshaped_strategies = strategy_data.reshape(100, -1, 300)
N_meaned_strategies = reshaped_strategies[:90].mean(axis=2).reshape(9, 10, 49).mean(axis=0)
N_meaned_decay_data = normalized_decay_curves.reshape(10, 10, 250, 25).mean(axis=(0, 2))

decay_rates = []
for i in N_meaned_decay_data:
    decay_rates.append(get_single_decay_rate(i, mut=0.5)[0])
decay_rates = np.array(decay_rates)

thresholds, base_chances = base_chance_threshold_fixed_prop([0, 0.19], 0.2, 7)
splits = [24, 20, 16, 12, 8, 4, 1]

base_chance_array  = np.array([base_chances] * 7).flatten()
splitting_array    = np.array([[24]*7, [20]*7, [16]*7, [12]*7, [8]*7, [4]*7, [1]*7]).flatten()

optimal_splits        = []
optimal_base_chances  = []
for i in range(10):
    max_val = N_meaned_strategies[i].argmax()
    optimal_splits.append(splitting_array[max_val])
    optimal_base_chances.append(base_chance_array[max_val])

# ---------------------------------------------------------------------------
# Step 4: predict strategies for every landscape × run (cell 94)
# ---------------------------------------------------------------------------

NKs_arr   = np.array(NKs)
k_over_ns = (NKs_arr[:, 1] + 1) / NKs_arr[:, 0]

predicted_base_chances = []
predicted_splittings   = []

for landscape in range(normalized_decay_curves.shape[0]):
    for run in range(normalized_decay_curves.shape[1]):
        run_data      = normalized_decay_curves[landscape, run, :]
        run_decay_rate = get_single_decay_rate(run_data, mut=0.5)[0]
        predicted_base_chances.append(
            optimal_base_chances[np.argmin(np.abs(decay_rates - run_decay_rate))])
        predicted_splittings.append(
            optimal_splits[np.argmin(np.abs(decay_rates - run_decay_rate))])

# ---------------------------------------------------------------------------
# Step 5: derive NK predictions (cell 98)
# ---------------------------------------------------------------------------

NK_samples       = [(45, 1), (45, 25), (45, 1), (45, 25)]
indexes_of_interest = [1900, 1400, 1000]
NK_bc_predictions  = [0.0, 0.0]
NK_th_predictions  = [0.8, 0.8]
NK_sp_predictions  = [1, 1]

for idx in indexes_of_interest:
    mean_bc_pred = np.array(predicted_base_chances[idx:idx+100]).mean()
    mean_sp_pred = np.array(predicted_splittings[idx:idx+100]).mean()
    bc = base_chances[np.argmin(np.abs(base_chances - mean_bc_pred))]
    th = thresholds[np.argmin(np.abs(base_chances - mean_bc_pred))]
    sp = splits[np.argmin(np.abs(np.array(splits) - mean_sp_pred))]
    NK_bc_predictions.append(bc)
    NK_th_predictions.append(th)
    NK_sp_predictions.append(sp)

# ---------------------------------------------------------------------------
# Step 6: run directed evolution (cells 100–101)
# ---------------------------------------------------------------------------

three_NK_results = []

for i in range(4):
    N, K  = NK_samples[i]
    bc    = NK_bc_predictions[i]
    s     = NK_sp_predictions[i]
    th    = NK_th_predictions[i]
    p     = 1200
    m     = 0.1

    print(f'Running DE: N={N}, K={K}, bc={bc}, split={s}')

    rep_rngs = jr.split(jr.PRNGKey(42), 100)

    def single_rep(rng_rep, s=s, bc=bc, th=th, N=N, K=K, p=p, m=m):
        split_rngs = jr.split(rng_rep, s)

        def single_s(s_rng):
            params = {'threshold': th, 'base_chance': bc}
            run = directedEvolution(s_rng,
                                    rng_rep,
                                    selection_strategy=slct.base_chance_threshold_select,
                                    selection_params=params,
                                    popsize=int(p // s),
                                    mut_chance=m / int(N),
                                    num_steps=50,
                                    num_reps=1,
                                    N=int(N),
                                    K=int(K),
                                    average=True)
            return run['fitness'][:, :, :].mean(axis=0)

        split_results = jax.vmap(single_s)(split_rngs)
        return jnp.array(split_results)

    repeat_results = jax.vmap(single_rep)(rep_rngs)
    three_NK_results.append(repeat_results)

# Extract winning splits (cell 101)
new_output = []
for res in three_NK_results:
    winning_splits_only = []
    for ii in range(res.shape[0]):
        rep         = res[ii]
        final_max   = rep[:, -1, :].max(axis=1)
        winning_split = np.argmax(final_max)
        winning_splits_only.append(rep[winning_split])
    new_output.append(np.array(winning_splits_only).mean(axis=(0, 2)))

# ---------------------------------------------------------------------------
# Step 7: save outputs (cells 102, 104)
# ---------------------------------------------------------------------------

with open(out_de, 'wb') as f:
    pickle.dump(new_output, f)
print(f'Saved → {out_de}')

with open(out_strat, 'wb') as f:
    pickle.dump((reshaped_strategies[19, :, :], reshaped_strategies[14, :, :]), f)
print(f'Saved → {out_strat}')
