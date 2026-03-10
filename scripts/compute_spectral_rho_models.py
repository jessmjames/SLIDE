"""
Compute spectral Dirichlet rho for fitness landscapes under different mutation models.

The "spectral rho" is the energy-weighted mean Laplacian eigenvalue in the
eigenbasis of the mutation operator.  For uniform AA mutations this reduces to
the existing get_dirichlet_metric formula; for nucleotide mutation models we
build the full nucleotide-space landscape (4^n_nuc points) and apply the
generalised Fourier transform using the eigenvectors of the per-site
transition matrix T.

Formula (nucleotide models):
    lambda_{k_1,...,k_n} = 1 - prod_j mu_{k_j}          (Laplacian eigenvalue)
    rho = sum_{k!=0} lambda_k * E_k  /  (d * sum_{k!=0} E_k)
    d   = n_nuc * (A_nuc - 1)  =  n_nuc * 3              (normalisation)

Usage
-----
    python scripts/compute_spectral_rho_models.py

Output
------
    Printed table + plot_data/spectral_rho_comparison.pkl
"""

import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import numpy as np
import pickle
from direvo_functions import CODON_MAPPER
from ruggedness_functions import get_dirichlet_metric
from slide_config import get_slide_data_dir

CODON_MAPPER_np = np.array(CODON_MAPPER, dtype=np.int32)   # (4,4,4) → AA or -1

landscape_dir  = os.path.join(parent_dir, 'landscape_arrays')
matrix_dir     = os.path.join(parent_dir, 'other_data')
plot_data_dir  = os.path.join(parent_dir, 'plot_data')
os.makedirs(plot_data_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# Load landscapes
# ---------------------------------------------------------------------------

def load_landscapes():
    names = ['GB1', 'TrpB', 'TEV', 'ParD3']
    files = ['GB1_landscape_array.pkl', 'TrpB_landscape_array.pkl',
             'TEV_landscape_array.pkl', 'E3_landscape_array.pkl']

    out = {}
    for name, fname in zip(names, files):
        with open(os.path.join(landscape_dir, fname), 'rb') as f:
            ld = np.array(pickle.load(f), dtype=np.float32)
        n_aa = ld.ndim          # infer from actual array shape
        out[name] = (ld, n_aa, n_aa * 3)   # (array, n_aa, n_nuc)
    return out


# ---------------------------------------------------------------------------
# Build nucleotide-space landscape
# ---------------------------------------------------------------------------

def build_nuc_landscape(landscape: np.ndarray, n_aa: int) -> np.ndarray:
    """
    Expand an AA landscape (shape (20,)*n_aa) to nucleotide space (shape (4,)*n_nuc).

    Each nucleotide-sequence is decoded by CODON_MAPPER; stop codons (-1) receive
    the minimum fitness value (via numpy's wrap-around indexing into buffered).
    """
    n_nuc   = n_aa * 3
    n_total = 4 ** n_nuc
    min_fit = float(landscape.min())

    # Pad so that index -1 wraps to the padded slot = min_fit.
    assert landscape.ndim == n_aa, f"landscape.ndim={landscape.ndim} != n_aa={n_aa}"
    buffered = np.pad(landscape, [(0, 1)] * n_aa, constant_values=min_fit)

    # Nucleotide index for position p in a sequence encoded as a base-4 integer:
    # digit_p = (linear_index >> 2*(n_nuc-1-p)) & 3
    idx = np.arange(n_total, dtype=np.int32)
    aa_indices = []
    for j in range(n_aa):
        shifts = [2 * (n_nuc - 1 - (3 * j + p)) for p in range(3)]
        d = [(idx >> np.int32(s)) & np.int32(3) for s in shifts]
        aa_j = CODON_MAPPER_np[d[0], d[1], d[2]]   # -1 for stop codons
        aa_indices.append(aa_j)
        del d

    f_flat = buffered[tuple(aa_indices)]
    return f_flat.reshape(tuple([4] * n_nuc)).astype(np.float64)


# ---------------------------------------------------------------------------
# Spectral rho  (generalised Dirichlet metric)
# ---------------------------------------------------------------------------

def spectral_rho_nuc(f_nuc: np.ndarray, T: np.ndarray) -> float:
    """
    Spectral Dirichlet rho for a nucleotide-space landscape under per-site
    transition matrix T (shape 4×4, rows sum to 1).

    Derivation
    ----------
    For a product mutation model (T applied independently at each site), the
    graph-Laplacian weight for pattern (k_1,...,k_n) in T's eigenbasis is:

        w_{k_1,...,k_n} = sum_j (1 - mu_{k_j})

    where mu_{k_j} is the k_j-th eigenvalue of T (sorted descending; mu_0 = 1).
    Trivial modes (k_j = 0) contribute 0.  The rho is:

        rho = sum_k w_k * E_k  /  (n_nuc * sum_k E_k)

    For the uniform A=4 case (mu_0=1, mu_{1,2,3} = -1/3) this reduces to
    get_dirichlet_metric(f_nuc), confirming the normalisation.

    Parameters
    ----------
    f_nuc : (4,)*n_nuc  float array
    T     : (4, 4) transition matrix (row-stochastic)

    Returns
    -------
    rho : float   (same scale as get_dirichlet_metric on AA landscape)
    """
    nuc_shape = f_nuc.shape
    n_nuc     = len(nuc_shape)

    f = f_nuc.astype(np.float64)
    f = f - f.mean()                     # remove DC; E_DC = 0 after this

    # Eigendecompose per-site T:  T = Q diag(mu) Q^{-1}
    eigenvalues, Q = np.linalg.eig(T)
    order       = np.argsort(-np.real(eigenvalues))   # descending (mu=1 first)
    eigenvalues = eigenvalues[order]
    Q           = Q[:, order]
    Q_inv       = np.linalg.inv(Q)

    # Generalised Fourier transform: f_hat = (Q_inv)^{⊗n_nuc} @ f
    f_hat = f.astype(np.complex128)
    for site in range(n_nuc):
        f_hat = np.tensordot(Q_inv, f_hat, axes=([1], [site]))
        f_hat = np.moveaxis(f_hat, 0, site)

    E = np.abs(f_hat) ** 2              # energy at each eigenbasis point

    # Laplacian weight: sum_j (1 - mu_{k_j})
    # Trivial mode k_j=0 → (1-mu_0)=0, contributes nothing automatically.
    one_minus_mu = np.real(1.0 - eigenvalues)          # shape (4,)
    L = np.zeros(nuc_shape, dtype=np.float64)
    for site in range(n_nuc):
        shape       = [1] * n_nuc
        shape[site] = 4
        L = L + one_minus_mu.reshape(shape)

    total_E = float(E.sum())
    if total_E < 1e-15:
        return 0.0

    rho = float(np.sum(L * E) / (n_nuc * total_E))
    return rho


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print("Loading landscapes...")
landscapes = load_landscapes()

print("Loading mutation matrices...")
h_sapiens_raw = np.load(os.path.join(matrix_dir, 'normed_h_sapiens_matrix.npy'))
e_coli_raw    = np.load(os.path.join(matrix_dir, 'normed_e_coli_matrix.npy'))

h_sapiens_sym = (h_sapiens_raw + h_sapiens_raw.T) / 2
h_sapiens_sym = h_sapiens_sym / h_sapiens_sym.sum(axis=1, keepdims=True)

# Uniform 4-state transition matrix: each site mutates to each of the other
# 3 states with equal probability (mu_0=1, mu_{1,2,3} = -1/3).
# This is the eigenbasis of the standard Fourier (DFT) transform on Z_4^n.
nuc_uniform = (np.ones((4, 4)) - np.eye(4)) / 3.0

# NOTE: spectral rho is only mathematically valid for SYMMETRIC T.
# For symmetric T, eigenvectors are orthonormal (spectral theorem), so the
# energy decomposition |f_hat_k|^2 is well-defined.
# For asymmetric T (e.g. nuc_e_coli), eigenvectors are not orthonormal and
# the Dirichlet form requires a π-weighted inner product — the formula below
# is not applicable.  nuc_e_coli is therefore excluded from spectral analysis.
MODELS = [
    ('aa_uniform',        None,          True),   # (name, T, is_symmetric)
    ('nuc_uniform',       nuc_uniform,   True),
    ('nuc_h_sapiens_sym', h_sapiens_sym, True),
    ('nuc_e_coli',        e_coli_raw,    False),  # asymmetric — spectral N/A
]

results = {}   # {landscape_name: {model_name: rho}}

for ld_name, (ld, n_aa, n_nuc) in landscapes.items():
    print(f"\n--- {ld_name} (n_aa={n_aa}, n_nuc={n_nuc}) ---")
    results[ld_name] = {}

    for model_name, T, is_sym in MODELS:
        if model_name == 'aa_uniform':
            rho = get_dirichlet_metric(ld)
            print(f"  {model_name:<22s} rho = {rho:.4f}  (AA Dirichlet metric)")
        elif not is_sym:
            rho = float('nan')
            print(f"  {model_name:<22s} rho = N/A   (asymmetric T — spectral not valid)")
        else:
            print(f"  {model_name:<22s}  building nuc landscape...", end=' ', flush=True)
            f_nuc = build_nuc_landscape(ld, n_aa)
            print(f"shape {f_nuc.shape}  computing rho...", end=' ', flush=True)
            rho = spectral_rho_nuc(f_nuc, T)
            print(f"rho = {rho:.4f}")

        results[ld_name][model_name] = rho

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

model_labels = [m for m, _, _ in MODELS]
print("\n\n=== Spectral rho summary ===\n")
header = f"{'Landscape':<10}" + "".join(f"{m:<24}" for m in model_labels)
print(header)
print("-" * len(header))
for ld_name in results:
    row = f"{ld_name:<10}"
    for m in model_labels:
        v = results[ld_name].get(m, float('nan'))
        row += f"{'N/A':<24}" if np.isnan(v) else f"{v:<24.4f}"
    print(row)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

out_path = os.path.join(plot_data_dir, 'spectral_rho_comparison.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(results, f)
print(f"\nSaved → {out_path}")
