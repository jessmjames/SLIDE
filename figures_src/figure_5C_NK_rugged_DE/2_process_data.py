"""
Figure 5C — Step 2: process NK directed evolution data (rugged, K=25).

Delegates entirely to figure_5B_NK_DE/2_process_data.py, which produces both
plot_data/NK_DE.pkl and plot_data/NK_strategy_spaces.pkl (shared by 5B and 5C).

Input
-----
  (same as Fig 5B — see figures_src/figure_5B_NK_DE/2_process_data.py)

Output
------
  plot_data/NK_DE.pkl          (shared with Fig 5B)
  plot_data/NK_strategy_spaces.pkl  (shared with Fig 5B)
"""

import subprocess
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
script = os.path.join(parent_dir, 'figures_src', 'figure_5B_NK_DE', '2_process_data.py')

result = subprocess.run([sys.executable, script], cwd=parent_dir)
sys.exit(result.returncode)
