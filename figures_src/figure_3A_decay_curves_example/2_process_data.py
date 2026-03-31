"""
Figure 3A — Step 2: fit decay rates to example curves.

Processing is done inline in ruggedness_figures_data_processing.ipynb (cell 8):
  1. Run directedEvolution() on NK (20,1) and NK (20,14)
  2. Normalise fitness trajectories to decay from 1
  3. Fit exponential decay rates via get_single_decay_rate()
  4. Compute fitted line arrays for plotting

Output
------
  plot_data/smooth_rugged_example.pkl
    (smooth_rugged, fitted_lines)
      smooth_rugged : array of shape (2, 25) — normalised mean fitness per generation
      fitted_lines  : array of shape (2, 25) — fitted exponential decay curves

This processing is not yet extracted into a standalone script.
Run cells 8–9 of ruggedness_figures_data_processing.ipynb to regenerate.
"""

import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
out = os.path.join(parent_dir, 'plot_data', 'smooth_rugged_example.pkl')

if os.path.exists(out):
    print(f"Already exists: {out}")
else:
    print("smooth_rugged_example.pkl not found.")
    print("Run cells 8–9 of ruggedness_figures_data_processing.ipynb to regenerate.")
