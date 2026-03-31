"""
Figure 5B — Step 2: process NK directed evolution data (smooth, K=1).

This step is performed inline in the data processing notebook
(ruggedness_figures_data_processing.ipynb) across several cells:

  Cell 98  — derive NK_bc_predictions, NK_sp_predictions, NK_th_predictions
              for NK_samples = [(45,1), (45,25), (45,1), (45,25)]
  Cell 99  — define modified directedEvolution function supporting splits
  Cell 100 — run DE simulations across 4 NK conditions (100 reps each)
  Cell 101 — extract winning splits in post-processing
  Cell 102 — save plot_data/NK_DE.pkl
  Cell 103 — (markdown)
  Cell 104 — save plot_data/NK_strategy_spaces.pkl
             (reshaped_strategies[19] = smooth K=1, reshaped_strategies[14] = rugged K=25)

Input
-----
  SLIDE_data/large_strategy_sweep.pkl
  SLIDE_data/large_decay_curve_sweep.pkl   (via upstream processing for predictions)

Output
------
  plot_data/NK_DE.pkl
    list of 4 arrays: [K1_baseline, K25_baseline, K1_SLIDE, K25_SLIDE]
    each array is the mean fitness trajectory over 100 reps

  plot_data/NK_strategy_spaces.pkl
    (smooth_strategies, rugged_strategies)
    smooth_strategies shape: (n_strategies, 300)
    rugged_strategies shape: (n_strategies, 300)

Usage
-----
  Run ruggedness_figures_data_processing.ipynb cells 98-104 in order.
  (No standalone script; processing is embedded in the notebook.)
"""

print("Processing is done in ruggedness_figures_data_processing.ipynb cells 98-104.")
print("Outputs:")
print("  plot_data/NK_DE.pkl")
print("  plot_data/NK_strategy_spaces.pkl")
