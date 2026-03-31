"""
Figure 3C — Step 2: fit decay rates across population sizes.

Processing is done in ruggedness_figures_data_processing.ipynb (cells 17–19):
  1. Reshape decay curves: (25, ?, 25) -> (25, 500, 25)
  2. Normalise each trajectory by its first timepoint
  3. Fit exponential decay rate rho for each (popsize, rep) pair via get_single_decay_rate()
  4. Population sizes are linspace(100, 2500, 25, dtype=int)

Input
-----
  SLIDE_data/popsize_accuracy.pkl  — raw decay curves

Output
------
  plot_data/popsize_accuracy.pkl
    (popsize_decay_rates, pops)
      popsize_decay_rates : array (25, 500) — fitted rho per replicate
      pops                : array (25,) — population sizes

Not yet extracted into a standalone script.
Run cells 17–19 of ruggedness_figures_data_processing.ipynb to regenerate.
"""

import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
out = os.path.join(parent_dir, 'plot_data', 'popsize_accuracy.pkl')

if os.path.exists(out):
    print(f"Already exists: {out}")
else:
    print("popsize_accuracy.pkl not found.")
    print("Run cells 17–19 of ruggedness_figures_data_processing.ipynb to regenerate.")
