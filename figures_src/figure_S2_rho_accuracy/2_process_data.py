"""
Figure S2 — Step 2: compute rho accuracy from NK codon sweeps.

Reads sweep pkl files from SLIDE_data/, fits decay rates, computes true
(K+1)/N for each NK combination, saves to plot_data/.

Input
-----
  SLIDE_data/large_decay_curve_sweep_codon_{e_coli,a_thaliana,human}.pkl

Output
------
  plot_data/ruggedness_accuracy_codon_{e_coli,a_thaliana,human}.pkl
  Each pkl: (k_plus_one_over_ns, decay_rates)
    k_plus_one_over_ns: (100,)   true (K+1)/N for each NK combo
    decay_rates:        (100, 250) estimated decay rates

Usage
-----
  python figures_src/figure_S2_rho_accuracy/2_process_data.py

Originally from scripts/ruggedness_accuracy_from_codon_sweeps.py on rebut_codon.
"""

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

import pickle
import numpy as np
import jax.numpy as jnp
from direvo_functions import get_single_decay_rate
from slide_config import get_slide_data_dir

slide_data_dir = str(get_slide_data_dir())
plot_data_dir  = os.path.join(parent_dir, 'plot_data')
os.makedirs(plot_data_dir, exist_ok=True)

MODELS   = ['e_coli', 'a_thaliana', 'human']
OVERWRITE = True


def NK_grid(N_range, num_samples=10):
    N = jnp.linspace(N_range[0], N_range[1], num=num_samples)
    K = jnp.array([jnp.linspace(1, i, num_samples) for i in N]).reshape(num_samples, num_samples)
    N = jnp.repeat(N, num_samples).reshape(num_samples, num_samples)
    return N, K


def compute_accuracy(decay_data, mut=0.5):
    reshaped   = decay_data.reshape(100, -1, 25)
    normalised = reshaped / reshaped[:, :, 0:1]

    decay_rates = np.zeros((normalised.shape[0], normalised.shape[1]))
    for i in range(normalised.shape[0]):
        for j in range(normalised.shape[1]):
            decay_rates[i, j] = get_single_decay_rate(normalised[i, j, :], mut=mut)[0]

    N_grid, K_grid = NK_grid([10, 50])
    Ns = jnp.flip(N_grid.flatten())
    Ks = jnp.flip(K_grid.flatten())
    NKs = list(zip(Ns, Ks))
    k_plus_one_over_ns = np.clip((np.array(NKs)[:, 1] + 1) / np.array(NKs)[:, 0], 0, 1)

    return k_plus_one_over_ns, decay_rates


if __name__ == '__main__':
    for label in MODELS:
        sweep_path = os.path.join(slide_data_dir, f'large_decay_curve_sweep_codon_{label}.pkl')
        if not os.path.exists(sweep_path):
            print(f'Missing: {sweep_path}')
            continue

        out_path = os.path.join(plot_data_dir, f'ruggedness_accuracy_codon_{label}.pkl')
        if (not OVERWRITE) and os.path.exists(out_path):
            print(f'Skipping existing: {out_path}')
            continue

        print(f'Processing {label}...')
        with open(sweep_path, 'rb') as f:
            decay_data = pickle.load(f)

        k_plus_one_over_ns, decay_rates = compute_accuracy(decay_data, mut=0.5)

        with open(out_path, 'wb') as f:
            pickle.dump((k_plus_one_over_ns, decay_rates), f)
        print(f'  Saved → {out_path}')
