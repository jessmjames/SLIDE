"""
Figure 5G — Step 1: raw simulation data for ParD3 directed evolution.

Runs three GPU simulation scripts.

Outputs (written to SLIDE_data/)
---------------------------------
  N3A20_decay_curves.pkl
    — decay curves for the 3-AA (ParD3) landscape
    — script: scripts/decay_curves_N3A20.py

  N3A20_strategy_sweep.pkl
    — strategy sweep for the 3-AA (ParD3) landscape; uses a 5x5 grid (N=3),
      distinct from the 7x7 grids used by Figs 5D–5F (N=4)
    — script: scripts/strategy_sweep_N3A20.py

  decay_curves_pard3_m0.1_multistart_10000_uniform.pkl
    — all-starts decay curves for ParD3 (and other empirical landscapes)
    — script: scripts/empirical_landscape_decay_curves.py

WARNING
-------
  strategy_sweep_E3_multistart_100_uniform_m0.025.pkl is required by the
  downstream data-processing step but has NO generation script on the main
  branch. This file must be present in SLIDE_data/ before proceeding to
  step 2. Contact the repository maintainer if it is missing.

Usage
-----
  python figures_src/figure_5G_ParD3_DE/1_raw_data.py

Note: requires a GPU.

Shared raw data: none — N3A20 outputs are unique to Fig 5G.
"""

import subprocess
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

scripts = [
    os.path.join(parent_dir, 'scripts', 'decay_curves_N3A20.py'),
    os.path.join(parent_dir, 'scripts', 'strategy_sweep_N3A20.py'),
    os.path.join(parent_dir, 'scripts', 'empirical_landscape_decay_curves.py'),
]

for script in scripts:
    print(f'\nRunning {os.path.basename(script)}...')
    result = subprocess.run([sys.executable, script], cwd=parent_dir)
    if result.returncode != 0:
        sys.exit(result.returncode)

print('\nDone.')
