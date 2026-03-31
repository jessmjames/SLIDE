"""
Figure 3D — Step 2: fit decay rates across mutation rates.

Processing is done in ruggedness_figures_data_processing.ipynb (cells 22–24):
  1. Reshape decay curves: (25, ?, 25) -> (25, 500, 25)
  2. Normalise each trajectory by its first timepoint
  3. Fit exponential decay rate rho for each (mut_rate, rep) pair via
     get_single_decay_rate(..., mut=muts[i])
  4. Mutation rates are linspace(0.01, 2, 25)

Input
-----
  SLIDE_data/mutation_rate_accuracy.pkl  — raw decay curves

Output
------
  plot_data/mut_accuracy.pkl
    (mut_decay_rates, muts)
      mut_decay_rates : array (25, 500) — fitted rho per replicate
      muts            : array (25,) — mutation rates

Not yet extracted into a standalone script.
Run cells 22–24 of ruggedness_figures_data_processing.ipynb to regenerate.
"""

import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
out = os.path.join(parent_dir, 'plot_data', 'mut_accuracy.pkl')

if os.path.exists(out):
    print(f"Already exists: {out}")
else:
    print("mut_accuracy.pkl not found.")
    print("Run cells 22–24 of ruggedness_figures_data_processing.ipynb to regenerate.")
