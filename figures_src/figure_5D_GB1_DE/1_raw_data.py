"""
Figure 5D — Step 1: raw simulation data for GB1 directed evolution.

Runs three GPU simulation scripts.

Outputs (written to SLIDE_data/)
---------------------------------
  N4A20_decay_curves.pkl
    — decay curves for all 4-AA landscapes (shared with Figs 5E, 5F)
    — script: scripts/decay_curves_N4A20.py

  N4A20_strategy_sweep.pkl
    — strategy sweep for all 4-AA landscapes (shared with Figs 5E, 5F)
    — script: scripts/strategy_sweep_N4A20.py

  decay_curves_gb1_m0.1_multistart_10000_uniform.pkl
    — all-starts decay curves for GB1 (and other empirical landscapes)
    — script: scripts/empirical_landscape_decay_curves.py

WARNING
-------
  strategy_sweep_GB1_multistart_100_uniform_m0.025.pkl is required by the
  downstream data-processing step but has NO generation script on the main
  branch. This file must be present in SLIDE_data/ before proceeding to
  step 2. Contact the repository maintainer if it is missing.

Usage
-----
  python figures_src/figure_5D_GB1_DE/1_raw_data.py

Note: requires a GPU.

Shared raw data: N4A20_decay_curves.pkl and N4A20_strategy_sweep.pkl are also
  used by Figs 5E and 5F.
"""

import subprocess
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

scripts = [
    os.path.join(parent_dir, 'scripts', 'decay_curves_N4A20.py'),
    os.path.join(parent_dir, 'scripts', 'strategy_sweep_N4A20.py'),
    os.path.join(parent_dir, 'scripts', 'empirical_landscape_decay_curves.py'),
]

for script in scripts:
    print(f'\nRunning {os.path.basename(script)}...')
    result = subprocess.run([sys.executable, script], cwd=parent_dir)
    if result.returncode != 0:
        sys.exit(result.returncode)

print('\nDone.')
