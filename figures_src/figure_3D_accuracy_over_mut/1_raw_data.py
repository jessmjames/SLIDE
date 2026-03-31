"""
Figure 3D — Step 1: generate decay curve sweep over mutation rates.

Raw simulation data is a sweep over 25 mutation rates (0.01 to 2.0) on a
fixed NK landscape, with 500 replicates each:

  SLIDE_data/mutation_rate_accuracy.pkl  — raw decay curves, shape (25, ?, 25)

This file is loaded in ruggedness_figures_data_processing.ipynb (cell 21).

No standalone script exists for this sweep — it was run interactively.
Contact the repository maintainer for details on how to re-run the sweep.
"""

import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
out = os.path.join(parent_dir, 'plot_data', 'mut_accuracy.pkl')

if os.path.exists(out):
    print(f"Already exists: {out}")
else:
    print("mut_accuracy.pkl not found.")
    print("Run cells 21–24 of ruggedness_figures_data_processing.ipynb to generate it.")
    print("Requires SLIDE_data/mutation_rate_accuracy.pkl.")
