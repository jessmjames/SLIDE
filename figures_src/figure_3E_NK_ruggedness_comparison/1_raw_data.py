"""
Figure 3E — Step 1: no raw data step.

All computation for this figure is self-contained (NK landscapes built analytically,
metrics computed on-the-fly). There are no external simulation files to generate.

Run Step 2 directly:
    python figures_src/figure_3E_NK_ruggedness_comparison/2_process_data.py
"""

import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
print("No raw data step for Fig 3E — all computation is in 2_process_data.py.")
print(f"Run: python {os.path.join(parent_dir, 'figures_src', 'figure_3E_NK_ruggedness_comparison', '2_process_data.py')}")
