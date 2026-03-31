"""
Figure 2D — Step 3: plot smooth landscape Fourier-space representation.

Input
-----
  No pkl files. All data computed inline.
  Projects smooth_func(x,y) = sin(y/3) onto the real orthonormal Fourier basis
  for N=4 and plots the 4×4 coefficient surface as a 3D figure.

Output
------
  figures/smooth_fourier_3D.pdf

Usage
-----
  python figures_src/figure_2D_smooth_fourier_3D/3_plot.py

Originally from ruggedness_figures_plots.ipynb cells 84, 85, 88, 89, 90.
"""

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

figures_dir = os.path.join(parent_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

# ---- Style settings (cell 84 / cell 90) ----
dpi = 350

# ---- Cell 84: grids and parameters ----
try:
    import jax.numpy as jnp
    xp = jnp
except Exception:
    xp = np

N = 4
fine_M = 101           # smooth surface sampling
BASE_Z = 0.0
GRID_COLOR_MAJOR = "0.35"
GRID_COLOR_MINOR = "0.65"
GRID_LW_MAJOR = 1.1
GRID_LW_MINOR = 0.6
GRID_ALPHA_MAJOR = 0.6
GRID_ALPHA_MINOR = 0.35

xs = np.arange(N)
X1d, X2d = np.meshgrid(xs, xs, indexing="ij")
t = np.linspace(0, N, fine_M, endpoint=False)
X1s, X2s = np.meshgrid(t, t, indexing="ij")

# ---- Cell 85: smooth function ----
def smooth_func(x, y):
    return xp.sin(y / 3)

# ---- Cell 88: build ordered basis ----
def theta(k1, k2, X1, X2):
    return (2 * xp.pi / N) * (k1 * X1 + k2 * X2)

def freq_mag(k):
    return int(min(k % N, (-k) % N))

SELF = {(0,0), (N//2,0), (0,N//2), (N//2,N//2)}

def is_rep(k1,k2):
    k1m, k2m = (-k1) % N, (-k2) % N
    if (k1,k2) == (k1m,k2m):
        return True
    return (k1,k2) < (k1m,k2m)

items = []
for k in [(0,0),(2,0),(0,2),(2,2)]:
    k1,k2 = k
    T_d = theta(k1,k2, xp.asarray(X1d), xp.asarray(X2d))
    T_s = theta(k1,k2, xp.asarray(X1s), xp.asarray(X2s))
    fd = xp.cos(T_d); fs = xp.cos(T_s)
    items.append(dict(
        k=(k1,k2), xmag=freq_mag(k1), ymag=freq_mag(k2),
        kind="cos", fd=np.array(fd), fs=np.array(fs),
        const_x=(freq_mag(k1)==0), const_y=(freq_mag(k2)==0)
    ))
for k1 in range(N):
    for k2 in range(N):
        if (k1,k2) in SELF:
            continue
        if not is_rep(k1,k2):
            continue
        T_d = theta(k1,k2, xp.asarray(X1d), xp.asarray(X2d))
        T_s = theta(k1,k2, xp.asarray(X1s), xp.asarray(X2s))
        for kind, trig in [("cos", xp.cos), ("sin", xp.sin)]:
            fd = xp.sqrt(2.0) * trig(T_d)
            fs = xp.sqrt(2.0) * trig(T_s)
            items.append(dict(
                k=(k1,k2), xmag=freq_mag(k1), ymag=freq_mag(k2),
                kind=kind, fd=np.array(fd), fs=np.array(fs),
                const_x=(freq_mag(k1)==0), const_y=(freq_mag(k2)==0)
            ))
assert len(items) == 16

def srt(it): return (it["xmag"], 0 if it["kind"]=="cos" else 1, it["k"])
rows = {0:[],1:[],2:[]}
for it in items:
    rows[it["ymag"]].append(it)

def take_first(pool, predicate, sort_key):
    cand = [it for it in pool if predicate(it)]
    cand.sort(key=sort_key)
    if not cand: return None
    pick = cand[0]; pool.remove(pick); return pick

pool0, pool1, pool2 = rows[0][:], rows[1][:], rows[2][:]
pool0.sort(key=srt); pool1.sort(key=srt); pool2.sort(key=srt)
grid = [[None]*4 for _ in range(4)]
grid[0][0] = take_first(pool0, lambda it: it["const_y"] and it["xmag"]==0, srt)
grid[0][1] = take_first(pool0, lambda it: it["const_y"] and it["xmag"]==1 and it["kind"]=="cos", srt)
grid[0][2] = take_first(pool0, lambda it: it["const_y"] and it["xmag"]==1 and it["kind"]=="sin", srt)
grid[0][3] = take_first(pool0, lambda it: it["const_y"] and it["xmag"]==2, srt)
grid[1][0] = take_first(pool1, lambda it: it["const_x"] and it["kind"]=="cos", srt)
grid[2][0] = take_first(pool1, lambda it: it["const_x"] and it["kind"]=="sin", srt)
for c in [1,2,3]:
    grid[1][c] = take_first(pool1, lambda it: not it["const_x"], srt)
for c in [1,2,3]:
    grid[2][c] = take_first(pool1, lambda it: not it["const_x"], srt)
grid[3][0] = take_first(pool2, lambda it: it["const_x"], srt)
grid[3][1] = take_first(pool2, lambda it: it["xmag"]==1 and it["kind"]=="cos", srt)
grid[3][2] = take_first(pool2, lambda it: it["xmag"]==1 and it["kind"]=="sin", srt)
grid[3][3] = take_first(pool2, lambda it: it["xmag"]==2, srt)
ordered = [grid[r][c] for r in range(4) for c in range(4)]
assert all(it is not None for it in ordered)
PERM = list(range(16))
ordered = [ordered[i] for i in PERM]

# ---- Cell 89: Fourier projection helpers ----
def project_to_fourier_coeffs(f_d, ordered, N):
    """
    Project a discrete function f_d (shape N×N) onto your real orthonormal basis
    defined by `ordered`. Returns a 4×4 array C whose (row, col) matches the
    panel order you're using (after PERM).
    """
    fvec = np.asarray(f_d, dtype=float).ravel()
    B = np.stack([it["fd"].ravel() for it in ordered], axis=1)  # (N^2 × 16)
    # Your basis columns are orthonormal w.r.t. (1/N^2) <.,.>, so:
    coeffs = (B.T @ fvec) / (N * N)  # (16,)
    C = coeffs.reshape(4, 4)         # row-major matches your panel order
    return C

def bilinear_surface_from_grid(C, Kx, Ky):
    """
    Bilinear interpolation over a 4×4 grid C onto fine frequency grid (Kx,Ky),
    with Kx,Ky in [0,3] along columns/rows respectively.
    Interpolates exactly at integer grid points.
    """
    # Clamp into valid cell range
    u = np.clip(Kx, 0.0, 3.0)
    v = np.clip(Ky, 0.0, 3.0)

    i0 = np.floor(v).astype(int)
    j0 = np.floor(u).astype(int)
    i1 = np.clip(i0 + 1, 0, 3)
    j1 = np.clip(j0 + 1, 0, 3)
    du = u - j0
    dv = v - i0

    # gather corners
    C00 = C[i0, j0]
    C10 = C[i1, j0]
    C01 = C[i0, j1]
    C11 = C[i1, j1]

    # bilinear blend
    S = ( (1 - du) * (1 - dv) * C00
        + (    du) * (1 - dv) * C01
        + (1 - du) * (    dv) * C10
        + (    du) * (    dv) * C11 )
    return S

def make_kspace_grids(fine_M=201):
    """
    Frequency-plane fine grid: continuous 'panel coordinates'.
    We use [0,3] because there are 4 columns (kx panels) and 4 rows (ky panels).
    """
    t = np.linspace(0.0, 3.0, fine_M)
    Kx, Ky = np.meshgrid(t, t, indexing="xy")
    return Kx, Ky

# ---- Cell 90: plot smooth Fourier space ----
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

# ---- Style settings ----
titlesize = 10
labelsize = 8
ticksize = 6
legendsize = 8
plt.rcParams["font.family"] = "DejaVu Sans"

# --- Fourier coefficients for smooth function ---
smooth_func_d = np.array(smooth_func(X1d, X2d))
C_smooth = project_to_fourier_coeffs(smooth_func_d, ordered, N)

# --- Fine Fourier grid ---
Kx, Ky = make_kspace_grids(fine_M=201)
Zs = bilinear_surface_from_grid(C_smooth, Kx, Ky)

# --- Plot ---
fig1 = plt.figure(figsize=(3, 3), dpi=300, constrained_layout=True)
ax1 = fig1.add_subplot(111, projection="3d")

norm = Normalize(vmin=Zs.min(), vmax=Zs.max())
colors = cm.viridis(norm(Zs))

# Smooth surface
ax1.plot_surface(Kx, Ky, Zs, facecolors=colors, linewidth=0, antialiased=True, alpha=1)

# # Coefficient crosses
# xs = np.arange(4)
# ys = np.arange(4)
# Xd, Yd = np.meshgrid(xs, ys, indexing="xy")
# Zd = C_smooth
# ax1.scatter(Xd.ravel(), Yd.ravel(), Zd.ravel(), marker='x', s=60,
#             depthshade=False, linewidths=1.1, color="gray")

# # Drop lines
# for xi, yi, zi in zip(Xd.ravel(), Yd.ravel(), Zd.ravel()):
#     ax1.plot([xi, xi], [yi, yi], [BASE_Z, zi], linewidth=2.0, alpha=0.9, color="gray")

# Title & fonts
ax1.set_title("Smooth Landscape Fourier Space", fontsize=titlesize, y=1)

# Axis labels
ax1.set_xlabel(r"$\hat{\sigma}_1$", fontsize=labelsize, labelpad=2)
ax1.set_ylabel(r"$\hat{\sigma}_2$", fontsize=labelsize, labelpad=2)
ax1.set_zlabel("Coefficient value", fontsize=labelsize)
ax1.zaxis.labelpad = 15

# Ticks
ax1.set_xticks(range(4))
ax1.set_yticks(range(4))
ax1.tick_params(axis="both", which="major", labelsize=ticksize)
ax1.tick_params(axis="both", which="minor", labelsize=ticksize)

out_path = os.path.join(figures_dir, 'smooth_fourier_3D.pdf')
plt.savefig(out_path, dpi=dpi)
print(f'Saved → {out_path}')
