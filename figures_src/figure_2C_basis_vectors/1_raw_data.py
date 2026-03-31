"""
Figure 2C — Step 1: raw data.

This figure requires no simulation data or pkl files.
All basis vectors are computed analytically from the real orthonormal Fourier
basis for an N=4 periodic grid.

The 16 basis functions are:
  - 4 self-conjugate cosine modes: cos[k1,k2] for (k1,k2) in {(0,0),(2,0),(0,2),(2,2)}
  - 12 paired modes: sqrt(2)*cos[k1,k2] and sqrt(2)*sin[k1,k2] for representative
    non-self-conjugate (k1,k2) pairs

All data is generated inline in 3_plot.py.

Originally from ruggedness_figures_plots.ipynb cell 88.
"""

print("No raw simulation data required for Figure 2C.")
print("All basis vectors are computed analytically — no pkl files needed.")
