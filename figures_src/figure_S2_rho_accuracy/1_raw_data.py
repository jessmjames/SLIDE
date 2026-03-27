"""
Figure S2 — Step 1: NK landscape sweep under codon mutation models.

Runs directed evolution over a grid of 100 NK landscapes (N∈[10,50], K∈[1,N])
for three organism mutation kernels, saving decay curve sweeps to SLIDE_data/.

Output
------
  SLIDE_data/large_decay_curve_sweep_codon_{e_coli,a_thaliana,human}.pkl

Each file shape: (100, 25, 10, 25)
  100 NK combinations × 25 reps × 10 seeds × 25 steps

Usage
-----
  python figures_src/figure_S2_rho_accuracy/1_raw_data.py

Note: requires a GPU. Originally from scripts/large_decay_curve_sweep_codon.py
on the rebut_codon branch.

Shared raw data: none (unique to S2).
"""

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

import pickle
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from direvo_functions import *
import selection_function_library as slct
import tqdm
from slide_config import get_slide_data_dir

slide_data_dir = str(get_slide_data_dir())

CODON_KERNELS = [
    ('e_coli',     os.path.join(parent_dir, 'other_data', 'normed_e_coli_matrix.npy')),
    ('a_thaliana', os.path.join(parent_dir, 'other_data', 'normed_a_thaliana_matrix.npy')),
    ('human',      os.path.join(parent_dir, 'other_data', 'normed_h_sapiens_matrix.npy')),
]

OVERWRITE = False
m   = 0.5
p   = 2500
REPS = 25


def NK_grid(N_range, num_samples=10):
    N = jnp.linspace(N_range[0], N_range[1], num=num_samples)
    K = jnp.array([jnp.linspace(1, i, num_samples) for i in N]).reshape(num_samples, num_samples)
    N = jnp.repeat(N, num_samples).reshape(num_samples, num_samples)
    return N, K


def directedEvolution(rng, selection_strategy, selection_params,
                      N=None, K=None, popsize=100, mut_chance=0.01,
                      num_steps=50, num_reps=10, mutation_matrix=None,
                      num_options=2):
    r1, r2, r3 = jr.split(rng, 3)
    i_pop             = jnp.array([jr.randint(r1, (N,), 0, num_options)] * popsize)
    fitness_function  = build_NK_landscape_function(r2, N, K)
    mutation_function = (build_mutation_function(mut_chance, num_options)
                         if mutation_matrix is None
                         else build_custom_mutation_function(mut_chance, mutation_matrix, A=num_options))
    selection_function = build_selection_function(selection_strategy, selection_params)

    pre_op_sel = build_selection_function(slct.base_chance_threshold_select,
                                          {'base_chance': 0.0, 'threshold': 0.95})
    pre_op = run_directed_evolution(r3, i_pop=i_pop,
                                    selection_function=pre_op_sel,
                                    mutation_function=mutation_function,
                                    fitness_function=fitness_function,
                                    num_steps=50)[1]
    i_pop = pre_op['pop'][-1]

    vmapped_run = jax.jit(jax.vmap(
        lambda r: run_directed_evolution(r, i_pop, selection_function,
                                         mutation_function, fitness_function=fitness_function,
                                         num_steps=num_steps)[1]
    ))
    results = vmapped_run(jr.split(r3, num_reps))
    return results


N_grid, K_grid = NK_grid([10, 50])
Ns = jnp.flip(N_grid.flatten())
Ks = jnp.flip(K_grid.flatten())
NKs = jnp.array(list(zip(Ns, Ks)))


def run_sweep(label, kernel):
    out_path = os.path.join(slide_data_dir, f'large_decay_curve_sweep_codon_{label}.pkl')
    if (not OVERWRITE) and os.path.exists(out_path):
        print(f'Skipping existing: {out_path}')
        return

    num_options = int(kernel.shape[0])
    sweep_results = np.zeros((100, REPS, 10, 25))

    with tqdm.tqdm(total=len(NKs), desc=label) as pbar:
        for i, nks in enumerate(NKs):
            rep_rngs = jr.split(jr.PRNGKey(42), REPS)

            def single_rep(rng_rep):
                params = {'threshold': 0.0, 'base_chance': 1.0}
                run = directedEvolution(rng_rep,
                                        N=int(nks[0]), K=int(nks[1]),
                                        selection_strategy=slct.base_chance_threshold_select,
                                        selection_params=params,
                                        popsize=int(p),
                                        mut_chance=m / int(nks[0]),
                                        num_steps=25, num_reps=10,
                                        mutation_matrix=kernel,
                                        num_options=num_options)
                return run['fitness'].mean(axis=-1)

            sweep_results[i] = np.array(jax.vmap(single_rep)(rep_rngs))
            pbar.update(1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(sweep_results, f)
    print(f'Saved {sweep_results.shape} → {out_path}')


if __name__ == '__main__':
    for label, path in CODON_KERNELS:
        kernel = np.load(path)
        run_sweep(label, kernel)
