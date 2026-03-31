"""
Figure 5A — Step 1: large NK strategy and decay-curve sweeps.

Runs two large GPU simulation sweeps over 100 NK landscapes that underpin
the strategy-prediction analysis.

Outputs (written to SLIDE_data/)
---------------------------------
  large_strategy_sweep_100.pkl   — strategy sweep over 100 NK landscapes
                                    shape: (100, 7, 7, 300)
                                    script: scripts/large_strategy_sweep.py

  large_decay_curve_sweep.pkl    — decay-curve sweep over 100 NK landscapes
                                    shape: (100, 25, 10, 25)
                                    script: scripts/large_decay_curve_sweep.py

Usage
-----
  python figures_src/figure_5A_strategy_prediction/1_raw_data.py

Note: requires a GPU. Both scripts are long-running.

Shared raw data: both outputs are also used by Figs 5B and 5C.
"""

import subprocess
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

scripts = [
    os.path.join(parent_dir, 'scripts', 'large_strategy_sweep.py'),
    os.path.join(parent_dir, 'scripts', 'large_decay_curve_sweep.py'),
]

for script in scripts:
    print(f'\nRunning {os.path.basename(script)}...')
    result = subprocess.run([sys.executable, script], cwd=parent_dir)
    if result.returncode != 0:
        sys.exit(result.returncode)

print('\nDone.')
