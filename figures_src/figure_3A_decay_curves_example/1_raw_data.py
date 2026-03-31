"""
Figure 3A — Step 1: generate example fitness decay curves for smooth and rugged NK.

Raw simulation data is produced inline in ruggedness_figures_data_processing.ipynb
(cell 8), which runs directed evolution on two NK landscapes:
  - N=20, K=1  (smooth, (K+1)/N = 0.1)
  - N=20, K=14 (rugged, (K+1)/N = 0.75)

No separate simulation script is needed — the directedEvolution() call is fast
(single replicate, 25 steps) and is run inside the data-processing notebook.

Also loaded in that notebook (cell 6):
  SLIDE_data/large_strategy_sweep.pkl
  SLIDE_data/large_decay_curve_sweep.pkl
(These are used for other sections, not for 3A specifically.)

To regenerate smooth_rugged_example.pkl, run cells 8–9 of
ruggedness_figures_data_processing.ipynb.
"""

import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
out = os.path.join(parent_dir, 'plot_data', 'smooth_rugged_example.pkl')

if os.path.exists(out):
    print(f"Already exists: {out}")
else:
    print("smooth_rugged_example.pkl not found.")
    print("Run cells 8–9 of ruggedness_figures_data_processing.ipynb to generate it.")
