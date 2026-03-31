"""
Figure 2A — Step 2: process data.

Computes the smooth landscape function on both the discrete 4×4 grid and the
fine 101×101 continuous grid used for surface plotting.

    smooth_func(x, y) = sin(y / 3)

No pkl output — all arrays are computed inline and used directly by 3_plot.py.

Originally from ruggedness_figures_plots.ipynb cells 84 and 85.
"""

import numpy as np

try:
    import jax.numpy as jnp
    xp = jnp
except Exception:
    xp = np

N = 4
fine_M = 101

xs = np.arange(N)
X1d, X2d = np.meshgrid(xs, xs, indexing="ij")
t = np.linspace(0, N, fine_M, endpoint=False)
X1s, X2s = np.meshgrid(t, t, indexing="ij")

def smooth_func(x, y):
    return xp.sin(y / 3)

smooth_func_s = smooth_func(X1s, X2s)
smooth_func_d = smooth_func(X1d, X2d)

if __name__ == "__main__":
    print(f"smooth_func_s shape: {np.array(smooth_func_s).shape}")
    print(f"smooth_func_d shape: {np.array(smooth_func_d).shape}")
    print("No pkl files written — arrays are computed inline.")
