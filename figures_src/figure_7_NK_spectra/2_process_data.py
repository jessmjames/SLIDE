"""
Figure 7 — Step 2: processing for NK Fourier spectra figure.

All processing for this figure is performed inline inside the plotting cell
(ruggedness_figures_plots.ipynb, cell 82). No intermediate pkl is written
between fourier_analysis.pkl and the final figure.

The processing steps performed in cell 82 are:
  1. Load NK landscape arrays from plot_data/fourier_analysis.pkl.
  2. For each landscape, compute the power spectrum via
     get_landscape_spectrum(f, norm=True, remove_constant=False, on_gpu=True)
     from ruggedness_functions.py.
  3. Compute the exponential-basis matrix via get_exp_matrix() (defined in
     cell 80 of the notebook).
  4. Compute fitness decay curves (dot product of exponential matrix and
     spectra) and per-term contributions.
  5. Fit a single decay rate rho via get_single_decay_rate() (cell 80).
  6. For two selected K values, also reconstruct the spectrum from noisy
     decay data via get_fourier_coeffs() with methods ls_constrained, nnls,
     and nnls_reg (defined in cell 80).

All helper functions (get_exp_matrix, get_fourier_coeffs, get_single_decay_rate)
are defined in cell 80 of ruggedness_figures_plots.ipynb and are NOT in any
standalone script.  They depend on:
  - numpy, scipy.optimize (curve_fit, lsq_linear, nnls)
  - ruggedness_functions.get_landscape_spectrum

Input
-----
  plot_data/fourier_analysis.pkl
    dict with keys: N_used, A_used, Ks_used, nk_builts

Output
------
  (No intermediate pkl is written — processing feeds directly into 3_plot.py)

To extract processing into a standalone script, copy helper function
definitions from cell 80 of ruggedness_figures_plots.ipynb.
"""

import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
inp = os.path.join(parent_dir, 'plot_data', 'fourier_analysis.pkl')

if os.path.exists(inp):
    print(f"Input exists: {inp}")
    print("No intermediate output file — processing is done inline in 3_plot.py.")
else:
    print("fourier_analysis.pkl not found.")
    print("Run 1_raw_data.py first (see its docstring).")
