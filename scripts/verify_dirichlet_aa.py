"""
Verify that spectral_rho_dirichlet (generalised Dirichlet form) matches
get_dirichlet_metric (FFT-based) on AA landscapes.

This is a sanity check for the Dirichlet formula before trusting the nuc
values. If they agree on AA, the formula is correct and any nuc discrepancy
lies elsewhere (codon mapping, mutation model, IK fitting, etc.).

Usage
-----
    python scripts/verify_dirichlet_aa.py

Output
------
    Printed table comparing the two methods for all 4 landscapes.
"""

import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import pickle
import numpy as np
from ruggedness_functions import get_dirichlet_metric
from scripts.compute_spectral_rho_models import spectral_rho_dirichlet

landscape_dir = os.path.join(parent_dir, 'landscape_arrays')

LANDSCAPES = {
    'GB1':   'GB1_landscape_array.pkl',
    'TrpB':  'TrpB_landscape_array.pkl',
    'TEV':   'TEV_landscape_array.pkl',
    'ParD3': 'E3_landscape_array.pkl',
}

A = 20
T_aa_uniform = (np.ones((A, A)) - np.eye(A)) / (A - 1)

print(f"{'Landscape':<10} {'FFT (get_dirichlet_metric)':<30} {'Dirichlet form (generalised)':<30} {'diff':<10}")
print("-" * 80)

for name, fname in LANDSCAPES.items():
    with open(os.path.join(landscape_dir, fname), 'rb') as f:
        ld = np.array(pickle.load(f), dtype=np.float64)

    rho_fft       = get_dirichlet_metric(ld)
    rho_dirichlet = spectral_rho_dirichlet(ld, T_aa_uniform)

    diff = rho_dirichlet - rho_fft
    print(f"{name:<10} {rho_fft:<30.6f} {rho_dirichlet:<30.6f} {diff:+.2e}")
