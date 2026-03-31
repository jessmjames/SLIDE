"""
Figure 2B — Step 1: raw data.

This figure requires no simulation data or pkl files.
The rugged landscape is defined analytically:

    bump_func(x, y) = -sin(x*pi/2 + y*pi/2) * cos(y*pi/4)

evaluated on a 4×4 discrete grid (N=4) and a fine continuous grid (101×101 points).
All data is generated inline in 2_process_data.py and 3_plot.py.

Originally from ruggedness_figures_plots.ipynb cells 84–85, 87.
"""

print("No raw simulation data required for Figure 2B.")
print("The rugged landscape is defined analytically — no pkl files needed.")
