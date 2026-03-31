"""
Figure 3B — Step 2: fit decay rates across the NK ruggedness grid.

Processing is done in ruggedness_figures_data_processing.ipynb (cells 11–14):
  1. Reshape decay curves: (100, ?, 25) -> (100, reps, 25)
  2. Normalise each trajectory by its first timepoint
  3. Fit exponential decay rate rho for each (NK, rep) pair via get_single_decay_rate()
  4. Compute (K+1)/N labels for the x-axis

Input
-----
  SLIDE_data/large_decay_curve_sweep.pkl  — raw decay curves

Output
------
  plot_data/ruggedness_accuracy.pkl
    (k_plus_one_over_ns, decay_rates)
      k_plus_one_over_ns : array (100,) — (K+1)/N for each NK pair
      decay_rates        : array (100, reps) — fitted rho per replicate

Not yet extracted into a standalone script.
Run cells 11–14 of ruggedness_figures_data_processing.ipynb to regenerate.
"""

import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
out = os.path.join(parent_dir, 'plot_data', 'ruggedness_accuracy.pkl')

if os.path.exists(out):
    print(f"Already exists: {out}")
else:
    print("ruggedness_accuracy.pkl not found.")
    print("Run cells 11–14 of ruggedness_figures_data_processing.ipynb to regenerate.")
