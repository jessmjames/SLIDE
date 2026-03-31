"""
Figure 5F — Step 2: process TEV directed evolution data.

This step is performed inline in the data processing notebook
(ruggedness_figures_data_processing.ipynb) across several cells:

  Cell 106 — load SLIDE_data/N4A20_strategy_sweep.pkl and N4A20_decay_curves.pkl
  Cell 107 — compute decay_rates, decay_means, optimal_pos from N4A20 sweeps
  Cell 108 — save plot_data/empirical_lookup.pkl
  Cell 109 — compute thresholds, base_chances, optimal_base_chances, optimal_splits
              for 7x7 grid (N=4): splits=[24,20,16,12,8,4,1]
  Cell 110 — load per-landscape SLIDE_data pkl files (decay curves + strategy sweeps)
              for GB1, TrpB, TEV, ParD3
  Cell 112 — define empirical_strategy_selection(), test_strategy_empirical(),
              uniform_start_locs() helper functions
  Cell 115 — run TEV strategy selection and DE test, save plot_data/TEV_strategy_selection.pkl

Input
-----
  SLIDE_data/N4A20_strategy_sweep.pkl
  SLIDE_data/N4A20_decay_curves.pkl
  SLIDE_data/decay_curves_tev_m0.1_multistart_10000_uniform.pkl
  SLIDE_data/strategy_sweep_TEV_multistart_100_uniform_m0.025.pkl
  landscape_arrays/TEV_landscape_array.pkl

Output
------
  plot_data/empirical_lookup.pkl
    (decay_rates, decay_means, optimal_pos, strategy_data_mean)
    — shared lookup table for Figs 5D, 5E, 5F; produced by cell 108

  plot_data/TEV_strategy_selection.pkl
    (x_vals, decay_mean, decay_rate, sweep, scipy_freq_matrix, run, scatter, line,
     TEV_decay_multi)

Usage
-----
  Run ruggedness_figures_data_processing.ipynb cells 106-115 in order.
  (No standalone script; processing is embedded in the notebook.)
"""

print("Processing is done in ruggedness_figures_data_processing.ipynb cells 106-115.")
print("Outputs:")
print("  plot_data/empirical_lookup.pkl  (shared with Figs 5D, 5E)")
print("  plot_data/TEV_strategy_selection.pkl")
