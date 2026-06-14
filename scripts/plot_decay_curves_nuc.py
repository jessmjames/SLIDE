"""
Plot 75-step decay curves for three nuc mutation models (supplementary figure).

3 rows (nuc_uniform, nuc_h_sapiens_sym, nuc_e_coli) × 4 landscapes.

Per panel:
  1. Data             — E[f̄(t)]² averaged over starting points, original fitness scale
  2. Idealised curve  — spectral ρ + true landscape mean c_true (solid)
  3. Fitted curve     — fitted ρ + fitted mean c_fit (dashed)
  Horizontal lines for true c and fitted c.

For nuc_e_coli (asymmetric), spectral ρ uses the symmetrised E. coli matrix
(nuc_e_coli_sym) for a well-defined energy decomposition, while c_true uses
the stationary distribution of the raw asymmetric matrix — a consistent pair.

c_true is computed using the correct stationary distribution for each mutation model:
  - uniform / symmetric → uniform (1/4 each nucleotide)
  - asymmetric (nuc_e_coli, nuc_h_sapiens) → eigenvector of transition matrix

Precomputed c_true values are cached in plot_data/true_constants_nuc.pkl
to avoid repeating the 64^4 = 16M-entry computation.

Usage
-----
  python scripts/plot_decay_curves_nuc.py

Output
------
  figures/decay_curves_nuc_75steps.pdf
"""

import sys
import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from direvo_functions import (
    get_single_decay_rate_IK_v2,
    model_function_IK_v2,
    CODON_MAPPER,
)
from slide_config import get_slide_data_dir

figures_dir   = os.path.join(parent_dir, 'figures')
plot_data_dir = os.path.join(parent_dir, 'plot_data')
landscape_dir = os.path.join(parent_dir, 'landscape_arrays')
matrix_dir    = os.path.join(parent_dir, 'other_data')
slide_data    = str(get_slide_data_dir())

os.makedirs(figures_dir, exist_ok=True)

_cm = np.array(CODON_MAPPER)  # (4, 4, 4)

# ---------------------------------------------------------------------------
# True asymptote helper — uses actual stationary distribution of mutation model
# ---------------------------------------------------------------------------

def stationary_dist(mut_matrix):
    """Per-nucleotide stationary distribution of a 4×4 row-stochastic matrix.
    Returns uniform [0.25]*4 for None (uniform model).
    """
    if mut_matrix is None:
        return np.ones(4) / 4.0
    M = np.array(mut_matrix, dtype=float)
    vals, vecs = np.linalg.eig(M.T)
    idx = np.argmin(np.abs(vals - 1.0))
    pi = np.real(vecs[:, idx])
    return pi / pi.sum()


def true_constants_nuc(ld_array, mut_matrix=None):
    """c_true = E_stat[f]² / E_AA[f²] using the correct stationary distribution.

    For uniform / symmetric models (mut_matrix=None), stationary dist = uniform.
    For asymmetric models, pass the 4×4 row-stochastic transition matrix.

    Stop codons → f_min (same as simulation).
    Returns (C_true, c_true) in normalised units.
    """
    f     = ld_array.astype(float)
    f_min = f.min()
    n_aa  = ld_array.ndim

    # Stationary distribution over single nucleotides
    pi = stationary_dist(mut_matrix)

    # Per-codon stationary weights: w[(i,j,k)] = pi[i]*pi[j]*pi[k]
    codon_weights = np.einsum('i,j,k->ijk', pi, pi, pi).reshape(-1)  # (64,)

    # Extended landscape: index 20 = stop codon → f_min
    f_ext = np.full((21,) * n_aa, f_min)
    f_ext[tuple([slice(20)] * n_aa)] = f

    # Map each codon index to AA index (or 20 for stop)
    aa_idx = np.where(_cm.reshape(-1) >= 0, _cm.reshape(-1), 20)  # (64,)

    # Full nuc-space fitness array: f_nuc[c1,c2,...] = f(aa1, aa2, ...)
    f_nuc = f_ext[np.ix_(*([aa_idx] * n_aa))]  # (64,)*n_aa

    # Full weight tensor: w_full[c1,c2,...] = codon_weights[c1] * ... * codon_weights[c_n]
    w_full = codon_weights
    for _ in range(n_aa - 1):
        w_full = np.multiply.outer(w_full, codon_weights)

    ef_stat = (w_full * f_nuc).sum()   # E_stat[f]
    ef2_aa  = np.mean(f ** 2)          # E_AA[f²] — starting distribution

    c = ef_stat ** 2 / ef2_aa
    return 1.0 - c, c


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

with open(os.path.join(plot_data_dir, 'spectral_rho_comparison.pkl'), 'rb') as f:
    spectral = pickle.load(f)

LANDSCAPE_FILES = {
    'GB1':   ('gb1',   os.path.join(landscape_dir, 'GB1_landscape_array.pkl')),
    'TrpB':  ('trpb',  os.path.join(landscape_dir, 'TrpB_landscape_array.pkl')),
    'TEV':   ('tev',   os.path.join(landscape_dir, 'TEV_landscape_array.pkl')),
    'ParD3': ('pard3', os.path.join(landscape_dir, 'E3_landscape_array.pkl')),
}

# Mutation matrices for asymmetric models (symmetric/uniform get None → uniform π)
e_coli_raw    = np.load(os.path.join(matrix_dir, 'normed_e_coli_matrix.npy'))

MODELS = [
    ('Uniform mutation',    'nuc_uniform',       'nuc_uniform',      None),
    ('Symmetric mutation',  'nuc_h_sapiens_sym', 'nuc_h_sapiens_sym', None),
    ('Asymmetric mutation', 'nuc_e_coli',        'nuc_e_coli_sym',    e_coli_raw),
]

# GB1=orange, TrpB=blue, TEV=green, ParD3=red
colours = [matplotlib.colormaps['tab10'](i) for i in [1, 0, 2, 3]]

# ---------------------------------------------------------------------------
# Pre-load landscape arrays
# ---------------------------------------------------------------------------

ld_arrays = {}
for ld_name, (_, ld_path) in LANDSCAPE_FILES.items():
    with open(ld_path, 'rb') as f:
        ld_arrays[ld_name] = pickle.load(f)

# ---------------------------------------------------------------------------
# Precompute / load cached c_true values
# ---------------------------------------------------------------------------

cache_path = os.path.join(plot_data_dir, 'true_constants_nuc.pkl')

if os.path.exists(cache_path):
    with open(cache_path, 'rb') as f:
        c_true_cache = pickle.load(f)
    print(f'Loaded c_true cache from {cache_path}')
else:
    c_true_cache = {}

cache_updated = False
for (_, _, _, mut_mat), model_suffix in zip(
        MODELS, [m[1] for m in MODELS]):
    for ld_name in LANDSCAPE_FILES:
        key = (ld_name, model_suffix)
        if key not in c_true_cache:
            print(f'  Computing c_true for {ld_name} × {model_suffix}...')
            C_n, c_n = true_constants_nuc(ld_arrays[ld_name], mut_mat)
            c_true_cache[key] = (C_n, c_n)
            cache_updated = True

# Fix: rebuild using correct model_suffix mapping
for model_name, model_suffix, spec_key, mut_mat in MODELS:
    for ld_name in LANDSCAPE_FILES:
        key = (ld_name, model_suffix)
        if key not in c_true_cache:
            print(f'  Computing c_true for {ld_name} × {model_suffix}...')
            C_n, c_n = true_constants_nuc(ld_arrays[ld_name], mut_mat)
            c_true_cache[key] = (C_n, c_n)
            cache_updated = True

if cache_updated:
    with open(cache_path, 'wb') as f:
        pickle.dump(c_true_cache, f)
    print(f'Saved c_true cache → {cache_path}')

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

NUM_STEPS = 75
steps     = np.arange(NUM_STEPS)
eps       = 1e-10

n_rows = len(MODELS)
n_cols = len(LANDSCAPE_FILES)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.8 * n_rows), dpi=300)

for row, (model_name, model_suffix, spec_key, _) in enumerate(MODELS):
    for col, (ld_name, (ld_key, _)) in enumerate(LANDSCAPE_FILES.items()):
        ax     = axes[row, col]
        colour = colours[col]

        fpath = os.path.join(slide_data,
                             f'decay_curves_{ld_key}_{model_suffix}_m0.1_all_starts_{NUM_STEPS}steps.pkl')
        with open(fpath, 'rb') as f:
            raw = pickle.load(f)

        # E[f̄(t)]² averaged over starting points, original fitness scale
        h     = raw.mean(axis=2).reshape(-1, NUM_STEPS)   # (N_starts, T)
        curve = (h ** 2).mean(axis=0)                      # (T,)
        scale = curve[0]

        # IK fit on normalised curve, scale back
        curve_norm = curve / (scale + eps)
        rho_raw, C_fit_n, c_fit_n = get_single_decay_rate_IK_v2(
            curve_norm, mut=0.1, num_steps=NUM_STEPS)
        C_fit = C_fit_n * scale
        c_fit = c_fit_n * scale

        # True constants from cache
        C_true_n, c_true_n = c_true_cache[(ld_name, model_suffix)]
        C_true = C_true_n * scale
        c_true = c_true_n * scale

        # Spectral ρ
        rho_spec = spectral[ld_name].get(spec_key, float('nan'))

        # Model curves in original scale
        fit_ik   = model_function_IK_v2(steps, rho_raw,      C_fit,  c_fit,  mut=0.1)
        fit_true = model_function_IK_v2(steps, rho_spec * 2, C_true, c_true, mut=0.1)

        ax.plot(steps, curve,    'k.', ms=2.5, alpha=0.5,            label='Data')
        ax.plot(steps, fit_true, '-',  color=colour, lw=1.5,         label='Idealised decay curve')
        ax.plot(steps, fit_ik,   '--', color=colour, lw=1.5, alpha=0.8,
                                                                       label='Fitted decay curve')
        ax.axhline(c_true, color=colour, ls=':',  lw=1.0, alpha=0.9, label='True $c$')
        ax.axhline(c_fit,  color=colour, ls='-.', lw=1.0, alpha=0.6, label='Fitted $c$')

        ax.set_xlim(0, NUM_STEPS - 1)
        ax.tick_params(labelsize=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if row == 0:
            ax.set_title(ld_name, fontsize=9)
        if col == 0:
            ax.set_ylabel(f'{model_name}\nFitness', fontsize=8, color='black')
        if row == n_rows - 1:
            ax.set_xlabel('Generations $M$', fontsize=8)

# Shared legend — dummy black/grey handles so colors don't bleed from the first panel
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
legend_handles = [
    mlines.Line2D([], [], color='k', marker='.', ms=5, ls='none',   label='Data'),
    mlines.Line2D([], [], color='k', lw=1.5,                        label='Idealised decay curve'),
    mlines.Line2D([], [], color='k', lw=1.5, ls='--', alpha=0.8,    label='Fitted decay curve'),
    mlines.Line2D([], [], color='k', lw=1.0, ls=':',  alpha=0.9,    label='True $c$'),
    mlines.Line2D([], [], color='k', lw=1.0, ls='-.', alpha=0.6,    label='Fitted $c$'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=5, fontsize=7,
           bbox_to_anchor=(0.5, -0.06))

plt.tight_layout()
out_path = os.path.join(figures_dir, f'decay_curves_nuc_{NUM_STEPS}steps.pdf')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f'Saved → {out_path}')
