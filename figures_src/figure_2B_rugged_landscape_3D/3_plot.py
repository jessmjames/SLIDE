"""
Figure 2B — Step 3: plot rugged 3D landscape.

Input
-----
  No pkl files. All data computed inline.
  bump_func(x, y) = -sin(x*pi/2 + y*pi/2) * cos(y*pi/4) evaluated on a 4×4
  discrete grid and a fine 101×101 grid (N=4, fine_M=101).

Output
------
  figures/rugged_landscape_3D.pdf

Usage
-----
  python figures_src/figure_2B_rugged_landscape_3D/3_plot.py

Originally from ruggedness_figures_plots.ipynb cells 84, 85, 87.
"""

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt

figures_dir = os.path.join(parent_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

# ---- Cell 84: grids and parameters ----
try:
    import jax.numpy as jnp
    xp = jnp
except Exception:
    xp = np

N = 4
fine_M = 101           # smooth surface sampling
BASE_Z = 0.0           # base plane/grid height

xs = np.arange(N)
X1d, X2d = np.meshgrid(xs, xs, indexing="ij")
t = np.linspace(0, N, fine_M, endpoint=False)
X1s, X2s = np.meshgrid(t, t, indexing="ij")

# ---- Cell 85: define functions ----
def bump_func(x, y):
    return -xp.sin(x * xp.pi / 2 + y * xp.pi / 2) * xp.cos(y * xp.pi / 4)

bump_func_s = bump_func(X1s, X2s)
bump_func_d = bump_func(X1d, X2d)

# ---- Cell 87: plot ----
import matplotlib.pyplot as plt

# ---- Style settings ----
titlesize = 10
labelsize = 8
ticksize = 6
legendsize = 8
dpi = 300
plt.rcParams["font.family"] = "DejaVu Sans"

cmap = plt.get_cmap("viridis")

# ---------- Second plot ----------
fig2 = plt.figure(figsize=(3, 3), dpi=300, constrained_layout=True)
ax2 = fig2.add_subplot(111, projection='3d')

# Normalize color scale
zmin = min(bump_func_d.min(), bump_func_s.min())
zmax = max(bump_func_d.max(), bump_func_s.max())
norm = plt.Normalize(zmin, zmax)

ax2.plot_surface(
    X1s, X2s, bump_func_s,
    linewidth=0, antialiased=True, alpha=1,   # match first plot's alpha
    cmap=cmap, norm=norm
)

# ax2.scatter(
#     X1d, X2d, bump_func_d,
#     marker='x', s=50, depthshade=False,
#     linewidths=1.5, color='gray'
# )

# for xi, yi, zi in zip(X1d.ravel(), X2d.ravel(), bump_func_d.ravel()):
#     ax2.plot([xi, xi], [yi, yi], [BASE_Z, zi], linewidth=2.0, alpha=0.9, color='gray')

# ---- Title & fonts ----
ax2.set_title("Rugged Landscape", fontsize=titlesize, y=1)

# ---- Axis label sizes ----
ax2.set_xlabel(r"$\sigma_1$", fontsize=labelsize, labelpad=2)
ax2.set_ylabel(r"$\sigma_2$", fontsize=labelsize, labelpad=2)
ax2.set_zlabel("Arbitrary Fitness", fontsize=labelsize)

# Adjust z-label position so it stays inside the figure
ax2.zaxis.labelpad = 15

# ---- Tick label sizes ----
ax2.tick_params(axis='both', which='major', labelsize=ticksize)
ax2.tick_params(axis='both', which='minor', labelsize=ticksize)

# Force integer ticks on all axes
from matplotlib.ticker import MaxNLocator
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

out_path = os.path.join(figures_dir, 'rugged_landscape_3D.pdf')
plt.savefig(out_path, dpi=dpi)
print(f'Saved → {out_path}')
