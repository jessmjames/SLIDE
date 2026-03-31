"""
Figure 3B — Step 1: generate large NK decay curve sweep across ruggedness.

Raw simulation data is a large sweep over NK landscapes on a 10x10 grid of
(N, K) values (N in [10, 50], K in [1, N]) with 10 samples each:

  SLIDE_data/large_decay_curve_sweep.pkl  — decay curve array, shape (100, ?, 25)
  SLIDE_data/large_strategy_sweep.pkl     — strategy sweep (not used for 3B)

These are loaded in ruggedness_figures_data_processing.ipynb (cell 6).
The NK grid is defined in cell 5 via NK_grid([10, 50], num_samples=10).

No standalone script exists for this sweep — it was run interactively.
Contact the repository maintainer for details on how to re-run the sweep.
"""

import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
out = os.path.join(parent_dir, 'plot_data', 'ruggedness_accuracy.pkl')

if os.path.exists(out):
    print(f"Already exists: {out}")
else:
    print("ruggedness_accuracy.pkl not found.")
    print("Run cells 5–14 of ruggedness_figures_data_processing.ipynb to generate it.")
    print("Requires SLIDE_data/large_decay_curve_sweep.pkl.")
