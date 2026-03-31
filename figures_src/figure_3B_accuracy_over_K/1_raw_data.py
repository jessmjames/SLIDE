"""
Figure 3B — Step 1: generate large NK decay curve sweep across ruggedness.

Runs a large sweep over NK landscapes on a 10x10 grid of (N, K) values
(N in [10, 50], K in [1, N]) with 10 samples each.

Output
------
  SLIDE_data/large_decay_curve_sweep.pkl  — decay curve array

Note: requires a GPU.
Shared raw data: also used by Figures 5A, 5B, and 5C.

Usage
-----
  python figures_src/figure_3B_accuracy_over_K/1_raw_data.py
"""

import subprocess
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

script = os.path.join(parent_dir, 'scripts', 'large_decay_curve_sweep.py')

print(f'Running {os.path.basename(script)}...')
result = subprocess.run([sys.executable, script], cwd=parent_dir)
if result.returncode != 0:
    sys.exit(result.returncode)

print('Done.')
