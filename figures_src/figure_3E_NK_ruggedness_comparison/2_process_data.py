"""
Figure 3E — Step 2: aggregate and normalise NK ruggedness metrics.

Processing is done in ruggedness_figures_data_processing.ipynb (cells 27–31):
  1. Reshape per-replicate metric arrays to (12, 50)
  2. Average over 50 replicates per K value
  3. Apply inverse transform to closest-max metric: 1 - 1/mean_NK_closest_max
  4. Min-max normalise all metrics to [0, 1] via norm_data()
  5. Compute k_over_ns = (K+1)/N for x-axis

Input
-----
  Computed in-memory from cells 26 (no external pkl file for raw metrics)

Output
------
  plot_data/NK_ruggedness_metric_comparison.pkl
    (mean_NK_roughness_to_slope, mean_NK_fourier, mean_NK_convergence_rates,
     mean_NK_paths_to_max, mean_NK_closest_max, k_over_ns, mean_NK_le_normed)
    All arrays shape (12,) — one value per K in [0, 11] for N=12.

Not yet extracted into a standalone script.
Run cells 27–31 of ruggedness_figures_data_processing.ipynb to regenerate.
"""

import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
out = os.path.join(parent_dir, 'plot_data', 'NK_ruggedness_metric_comparison.pkl')

if os.path.exists(out):
    print(f"Already exists: {out}")
else:
    print("NK_ruggedness_metric_comparison.pkl not found.")
    print("Run cells 27–31 of ruggedness_figures_data_processing.ipynb to regenerate.")
