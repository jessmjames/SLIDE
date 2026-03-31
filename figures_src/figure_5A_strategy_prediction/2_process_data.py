"""
Figure 5A — Step 2: process strategy sweep data.

Delegates to scripts/plot_strategy_prediction.py, which loads the raw
simulation data, computes optimal strategies and prediction accuracy, saves
two intermediate pkl files, and then produces the figure.

Input
-----
  SLIDE_data/large_strategy_sweep_100.pkl
  SLIDE_data/large_decay_curve_sweep.pkl

Output
------
  plot_data/optimal_DE_strategies.pkl
    (decay_rates, optimal_splits, optimal_base_chances)

  plot_data/strategy_prediction_accuracy.pkl
    (actual_k_over_ns, bc_means, bc_stds, sp_means, sp_stds)

Usage
-----
  python figures_src/figure_5A_strategy_prediction/2_process_data.py
"""

import subprocess
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
script = os.path.join(parent_dir, 'scripts', 'plot_strategy_prediction.py')

result = subprocess.run([sys.executable, script], check=True)
sys.exit(result.returncode)
