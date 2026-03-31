"""
Figure 5G — Step 2: process ParD3 directed evolution data.

ParD3 uses a 5x5 strategy grid (N=3, splits=[20,15,10,5,1]) rather than the 7x7
grid (N=4) used by GB1, TrpB, and TEV. It is processed with a separate sweep.

This step is performed inline in the data processing notebook
(ruggedness_figures_data_processing.ipynb) across several cells:

  Cell 116 — load SLIDE_data/N3A20_strategy_sweep.pkl and N3A20_decay_curves.pkl
  Cell 117 — compute decay_rates, decay_means, optimal_pos from N3A20 sweeps
  Cell 118 — compute thresholds, base_chances for 5x5 grid
  Cell 119 — run ParD3 strategy selection and DE test,
              save plot_data/ParD3_strategy_selection.pkl
              (uses empirical_strategy_selection(..., N=3) and test_strategy_empirical())

Input
-----
  SLIDE_data/N3A20_strategy_sweep.pkl
  SLIDE_data/N3A20_decay_curves.pkl
  SLIDE_data/decay_curves_pard3_m0.1_multistart_10000_uniform.pkl
  SLIDE_data/strategy_sweep_E3_multistart_100_uniform_m0.025.pkl
  landscape_arrays/ParD3_landscape_array.pkl

Output
------
  plot_data/ParD3_strategy_selection.pkl
    (x_vals, decay_mean, decay_rate, sweep, scipy_freq_matrix, run, scatter, line,
     ParD3_decay_multi)

Usage
-----
  Run ruggedness_figures_data_processing.ipynb cells 116-119 in order.
  (No standalone script; processing is embedded in the notebook.)
"""

print("Processing is done in ruggedness_figures_data_processing.ipynb cells 116-119.")
print("Output:")
print("  plot_data/ParD3_strategy_selection.pkl")
