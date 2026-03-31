"""
Figure 7 — Step 1: NK landscape sweep data for Fourier spectra.

The plot_data input for this figure is:

  plot_data/fourier_analysis.pkl

This pickle contains a dict with keys:
  'N_used'     : int   — gene length N (e.g. 10)
  'A_used'     : int   — number of alleles A (e.g. 20)
  'Ks_used'    : list  — K values swept (e.g. [0, 1, 3, 5, 9])
  'nk_builts'  : list  — one NK landscape array per K value,
                         each with shape (A,)*N built via
                         get_nk_l_o_shape(rng, N, K, shape=(A,)*N)
                         from ruggedness_functions.py

The landscapes are constructed analytically (no simulation required):
get_nk_l_o_shape calls build_NK_landscape_function (direvo_functions.py)
and evaluates it on the full (A^N) genotype space.

No standalone script currently exists for building fourier_analysis.pkl.
It was created interactively (origin notebook/script unknown).

To regenerate, build NK landscapes for a chosen set of Ks using:

  from ruggedness_functions import get_nk_l_o_shape
  import jax.random as jr, pickle

  N, A = 10, 20
  Ks = [0, 1, 3, 5, 9]
  shape = (A,) * N
  rng = jr.PRNGKey(0)
  nk_builts = [get_nk_l_o_shape(jr.fold_in(rng, K), N, K, shape) for K in Ks]

  with open('plot_data/fourier_analysis.pkl', 'wb') as f:
      pickle.dump({'N_used': N, 'A_used': A, 'Ks_used': Ks, 'nk_builts': nk_builts}, f)

Note: building A^N landscapes is memory-intensive. For N=10, A=20 that is
20^10 ≈ 10^13 entries — you may need to reduce N or A for testing.
Contact the repository maintainer for the original parameter choices.
"""

import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
out = os.path.join(parent_dir, 'plot_data', 'fourier_analysis.pkl')

if os.path.exists(out):
    print(f"Already exists: {out}")
else:
    print("fourier_analysis.pkl not found.")
    print("See docstring above for how to regenerate it.")
