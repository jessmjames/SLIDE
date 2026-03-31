"""
Figure 5D — Step 2: build N4A20 empirical lookup table and run GB1 directed evolution.

Pipeline (notebook cells 106–113):
  1. Load N4A20_strategy_sweep.pkl + N4A20_decay_curves.pkl.
  2. Compute decay_rates, decay_means, and optimal strategy positions (cell 107).
  3. Save plot_data/empirical_lookup.pkl — shared by Figs 5D, 5E, 5F (cell 108).
  4. Compute thresholds/base_chances/splits grid (cell 109).
  5. Load GB1 decay curves, strategy sweep, and landscape array (cells 110, 34).
  6. Run empirical_strategy_selection() → predict optimal strategy from GB1 decay.
  7. Run test_strategy_empirical() — 100 reps × 2 strategies (baseline + SLIDE)
     from 10 uniformly spaced starting locations (cell 113).
  8. Save plot_data/GB1_strategy_selection.pkl.

WARNING: Step 7 is GPU-intensive (100 reps × 2 strategies × 10 starts).
⚠️  strategy_sweep_GB1_multistart_100_uniform_m0.025.pkl has no generation
    script on main — see 1_raw_data.py for details.

Input
-----
  SLIDE_data/N4A20_strategy_sweep.pkl
  SLIDE_data/N4A20_decay_curves.pkl
  SLIDE_data/decay_curves_gb1_m0.1_multistart_10000_uniform.pkl
  SLIDE_data/strategy_sweep_GB1_multistart_100_uniform_m0.025.pkl
  landscape_arrays/GB1_landscape_array.pkl

Output
------
  plot_data/empirical_lookup.pkl
    (decay_rates, decay_means, optimal_pos, strategy_data_mean)
    — shared lookup for Figs 5D, 5E, 5F

  plot_data/GB1_strategy_selection.pkl
    (x_vals, decay_mean, decay_rate, sweep, scipy_freq_matrix, run,
     scatter, line, GB1_decay_multi)

Source
------
  ruggedness_figures_data_processing.ipynb, cells 34, 106–113.
"""

import os
import sys
import pickle
import numpy as np
import tqdm

import jax
import jax.numpy as jnp
import jax.random as jr

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

from direvo_functions import (
    build_empirical_landscape_function,
    build_mutation_function,
    build_selection_function,
    run_directed_evolution,
    base_chance_threshold_fixed_prop,
    get_single_decay_rate,
    model_function,
)
import selection_function_library as slct
from slide_config import get_slide_data_dir

slide_data_dir   = get_slide_data_dir()
plot_data_dir    = os.path.join(parent_dir, 'plot_data')
landscape_dir    = os.path.join(parent_dir, 'landscape_arrays')
os.makedirs(plot_data_dir, exist_ok=True)

out_lookup = os.path.join(plot_data_dir, 'empirical_lookup.pkl')
out_gb1    = os.path.join(plot_data_dir, 'GB1_strategy_selection.pkl')

# ---------------------------------------------------------------------------
# Modified directedEvolution supporting splits (notebook cell 99)
# ---------------------------------------------------------------------------

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
        fitness_function  = build_empirical_landscape_function(landscape)
        mutation_function = build_mutation_function(mut_chance, 20)
    else:
        from direvo_functions import build_NK_landscape_function
        fitness_function  = build_NK_landscape_function(rng_rep, N, K)
        mutation_function = build_mutation_function(mut_chance, 2)

    selection_function = build_selection_function(selection_strategy, selection_params)

    vmapped_run = jax.jit(jax.vmap(lambda r: run_directed_evolution(
        r, i_pop, selection_function, mutation_function,
        fitness_function=fitness_function, num_steps=num_steps)[1]))

    rng_seeds = jr.split(s_rng, num_reps)
    return vmapped_run(rng_seeds)

# ---------------------------------------------------------------------------
# Helper functions (notebook cell 112)
# ---------------------------------------------------------------------------

def empirical_strategy_selection(decay, sweep, decay_rates, optimal_base_chances,
                                  optimal_splits, N=4):

    def strategy_from_decay(decay_rate, standard_decay_rates=decay_rates,
                             optimal_base_chances=optimal_base_chances,
                             optimal_splits=optimal_splits):
        optimum_index = np.argmin(np.abs(standard_decay_rates - decay_rate))
        return optimal_base_chances[optimum_index], optimal_splits[optimum_index]

    decay_mean = (decay ** 2).mean(axis=(0, 1, 2))
    decay_std  = (decay ** 2).std(axis=(0, 1, 2))
    decay_mean = decay_mean / decay_mean[0]
    decay_rate = get_single_decay_rate(decay_mean, mut=0.1)
    x_vals     = np.linspace(0, 24, 25)
    strategy   = strategy_from_decay(decay_rate[0] / 2)

    print(decay_rate[0] / 2)

    def normalise_decay(y_vals, constant):
        out = y_vals - constant
        return out / out[0]

    scatter = normalise_decay(decay_mean, decay_rate[1])
    line    = normalise_decay(model_function(x_vals, *decay_rate), decay_rate[1])

    if N == 3:
        scipy_freq_matrix = np.zeros((5, 5), dtype=int)
        _, bc_arr = base_chance_threshold_fixed_prop([0, 0.19], 0.2, 5)
        sp_arr    = [20, 15, 10, 5, 1]
    else:
        scipy_freq_matrix = np.zeros((7, 7), dtype=int)
        _, bc_arr = base_chance_threshold_fixed_prop([0, 0.19], 0.2, 7)
        sp_arr    = [24, 20, 16, 12, 8, 4, 1]

    j = np.where(np.array(bc_arr) == strategy[0])[0]
    i = np.where(np.array(sp_arr) == strategy[1])[0]
    scipy_freq_matrix[i, j] = 1

    print(strategy)
    return (x_vals, decay_mean, decay_rate, sweep, scipy_freq_matrix, strategy, scatter, line)


def uniform_start_locs(ld, num=10000):
    flat_ld     = ld.flatten()
    flat_indexes = np.round(np.linspace(0, flat_ld.shape[0] - 1, num)).astype(int)
    indexes     = np.array([np.unravel_index(i, ld.shape) for i in flat_indexes])
    return indexes


def test_strategy_empirical(ld, bcs, sps, ths, starts):
    ld = jnp.array(ld)
    start_results = []

    for start in tqdm.tqdm(starts):
        results = []

        for i in range(2):
            bc = bcs[i]
            s  = sps[i]
            th = ths[i]
            p  = 1200 if len(ld.shape) == 4 else 60
            m  = 0.01

            rep_rngs = jr.split(jr.PRNGKey(42), 100)

            def single_rep(rng_rep, s=s, bc=bc, th=th, p=p, m=m, start=start):
                split_rngs = jr.split(rng_rep, s)

                def single_s(s_rng):
                    params = {'threshold': th, 'base_chance': bc}
                    run = directedEvolution(s_rng,
                                            rng_rep,
                                            selection_strategy=slct.base_chance_threshold_select,
                                            selection_params=params,
                                            popsize=int(p // s),
                                            mut_chance=m,
                                            num_steps=150,
                                            num_reps=1,
                                            define_i_pop=jnp.array([start] * int(p // s)),
                                            empirical=True,
                                            landscape=ld,
                                            average=True)
                    return run['fitness'][:, :, :].mean(axis=0)

                split_results = jax.vmap(single_s)(split_rngs)
                return jnp.array(split_results)

            repeat_results = jax.vmap(single_rep)(rep_rngs)
            results.append(repeat_results)

        # Extract winning splits
        run = []
        for res in results:
            winning_splits_only = []
            for ii in range(res.shape[0]):
                rep           = res[ii]
                final_max     = rep[:, -1, :].max(axis=1)
                winning_split = np.argmax(final_max)
                winning_splits_only.append(rep[winning_split])
            run.append(np.array(winning_splits_only))

        start_results.append(run)

    return start_results

# ---------------------------------------------------------------------------
# Step 1: build N4A20 empirical lookup (cells 106–109)
# ---------------------------------------------------------------------------

if not os.path.exists(out_lookup):
    print('Building empirical lookup from N4A20 data ...')

    with open(os.path.join(slide_data_dir, 'N4A20_strategy_sweep.pkl'), 'rb') as f:
        strategy_data = pickle.load(f)
    with open(os.path.join(slide_data_dir, 'N4A20_decay_curves.pkl'), 'rb') as f:
        decay_data = pickle.load(f)

    decay_rates_lookup  = []
    decay_means_lookup  = []
    for n, i in enumerate(decay_data[1:]):
        decay_mean = (i ** 2).mean(axis=(0, 1, 2))
        decay_mean = decay_mean / decay_mean[0]
        decay_means_lookup.append(decay_mean)
        decay_rate = get_single_decay_rate(decay_mean, mut=0.1)
        decay_rates_lookup.append(decay_rate[0] / 2)

    optimal_pos = []
    for n, i in enumerate(np.array(strategy_data).mean(axis=0)):
        max_pos = np.unravel_index(np.argmax(i.T), i.shape)
        optimal_pos.append(max_pos)

    with open(out_lookup, 'wb') as f:
        pickle.dump((decay_rates_lookup, decay_means_lookup,
                     optimal_pos, np.array(strategy_data).mean(axis=0)), f)
    print(f'Saved → {out_lookup}')
else:
    print(f'Using existing {out_lookup}')
    with open(out_lookup, 'rb') as f:
        decay_rates_lookup, decay_means_lookup, optimal_pos, _ = pickle.load(f)

# Derive optimal strategies from lookup (cell 109)
thresholds_lk, base_chances_lk = base_chance_threshold_fixed_prop([0, 0.19], 0.2, 7)
splits_lk = [24, 20, 16, 12, 8, 4, 1]
optimal_base_chances_lk = [base_chances_lk[i[1]] for i in optimal_pos]
optimal_splits_lk       = [splits_lk[i[0]]       for i in optimal_pos]

# ---------------------------------------------------------------------------
# Step 2: GB1 — load data (cells 110, 34)
# ---------------------------------------------------------------------------

if os.path.exists(out_gb1):
    print(f'Already exists: {out_gb1}')
    sys.exit(0)

with open(os.path.join(slide_data_dir, 'decay_curves_gb1_m0.1_multistart_10000_uniform.pkl'), 'rb') as f:
    GB1_decay_multi = pickle.load(f)
with open(os.path.join(slide_data_dir, 'strategy_sweep_GB1_multistart_100_uniform_m0.025.pkl'), 'rb') as f:
    GB1_sweep_multi = pickle.load(f)
with open(os.path.join(landscape_dir, 'GB1_landscape_array.pkl'), 'rb') as f:
    GB1 = pickle.load(f)

# ---------------------------------------------------------------------------
# Step 3: strategy selection + DE (cell 113)
# ---------------------------------------------------------------------------

x_vals, decay_mean, decay_rate, sweep, scipy_freq_matrix, strategy, scatter, line = \
    empirical_strategy_selection(GB1_decay_multi, GB1_sweep_multi,
                                 decay_rates_lookup, optimal_base_chances_lk, optimal_splits_lk)

run = test_strategy_empirical(
    GB1,
    [0.0, strategy[0]],
    [1,   strategy[1]],
    [0.8, thresholds_lk[np.array(base_chances_lk) == strategy[0]]],
    starts=uniform_start_locs(ld=GB1, num=10),
)

with open(out_gb1, 'wb') as f:
    pickle.dump((x_vals, decay_mean, decay_rate, sweep, scipy_freq_matrix,
                 run, scatter, line, GB1_decay_multi), f)
print(f'Saved → {out_gb1}')
