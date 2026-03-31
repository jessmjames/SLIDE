"""
Figure 3A — Step 1: no raw data step.

All computation for this figure is self-contained (runs directedEvolution inline
on two NK landscapes; no external simulation files needed).

Run Step 2 directly:
    python figures_src/figure_3A_decay_curves_example/2_process_data.py
"""

import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
print("No raw data step for Fig 3A — all computation is in 2_process_data.py.")
print(f"Run: python {os.path.join(parent_dir, 'figures_src', 'figure_3A_decay_curves_example', '2_process_data.py')}")
