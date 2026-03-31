"""
Figure 5C — Step 1: raw simulation data for NK directed evolution (rugged, K=25).

Raw data is shared with Fig 5A. This script delegates entirely to Fig 5A's
1_raw_data.py, which runs the two large GPU sweeps:

  SLIDE_data/large_strategy_sweep_100.pkl  (scripts/large_strategy_sweep.py)
  SLIDE_data/large_decay_curve_sweep.pkl   (scripts/large_decay_curve_sweep.py)

Usage
-----
  python figures_src/figure_5C_NK_rugged_DE/1_raw_data.py

Note: requires a GPU (via the delegated 5A script).

Shared raw data: outputs are also used by Figs 5A and 5B.
"""

import subprocess
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

script_5a = os.path.join(parent_dir, 'figures_src', 'figure_5A_strategy_prediction', '1_raw_data.py')

print('Delegating to Fig 5A 1_raw_data.py...')
result = subprocess.run([sys.executable, script_5a], cwd=parent_dir)
if result.returncode != 0:
    sys.exit(result.returncode)

print('\nDone.')
