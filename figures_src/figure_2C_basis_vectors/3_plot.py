"""
Figure 2C — Step 3: plot real orthonormal Fourier basis vectors (4×4 grid).

Input
-----
  No pkl files. All data computed inline.
  Constructs 16 real orthonormal Fourier basis functions for an N=4 periodic grid.

Output
------
  figures/basis_vectors.pdf

Usage
-----
  python figures_src/figure_2C_basis_vectors/3_plot.py

Originally from ruggedness_figures_plots.ipynb cell 88.
"""

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

figures_dir = os.path.join(parent_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt

## Fontsizes

titlesize = 10
labelsize = 8
ticksize = 6
legendsize = 8
dpi = 350
plt.rcParams["font.family"] = "DejaVu Sans"
c2 = 'tab:blue'#298c8c'
c1 = 'tab:orange' #800074'
c3 = '#f55f74'
c4 = 'tab:green'

# Optional JAX acceleration
try:
    import jax.numpy as jnp
    xp = jnp
except Exception:
    xp = np

N = 4
fine_M = 101           # smooth surface sampling
BASE_Z = 0.0           # base plane/grid height
SHOW_BASE_PLANE = False
SHOW_BASE_GRID = True
GRID_MAJOR_STEP = 1.0  # draw lines every 1.0 unit (integer grid)
GRID_MINOR_STEP = 0.5  # set to None to disable minor grid
GRID_COLOR_MAJOR = "0.35"
GRID_COLOR_MINOR = "0.65"
GRID_LW_MAJOR = 1.1
GRID_LW_MINOR = 0.6
GRID_ALPHA_MAJOR = 0.6
GRID_ALPHA_MINOR = 0.35

# --- grids ---
xs = np.arange(N)
X1d, X2d = np.meshgrid(xs, xs, indexing="ij")
t = np.linspace(0, N, fine_M, endpoint=False)
X1s, X2s = np.meshgrid(t, t, indexing="ij")

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

# 1) self-conjugate cos modes
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

# 2) paired reps: √2·cos and √2·sin
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

# --- Construct ordered 4x4 grid with constraints -----------------------------
def srt(it): return (it["xmag"], 0 if it["kind"]=="cos" else 1, it["k"])

rows = {0:[],1:[],2:[]}
for it in items:
    rows[it["ymag"]].append(it)

def take_first(pool, predicate, sort_key):
    cand = [it for it in pool if predicate(it)]
    cand.sort(key=sort_key)
    if not cand:
        return None
    pick = cand[0]
    pool.remove(pick)
    return pick

pool0, pool1, pool2 = rows[0][:], rows[1][:], rows[2][:]
pool0.sort(key=srt); pool1.sort(key=srt); pool2.sort(key=srt)

grid = [[None]*4 for _ in range(4)]
# Row 0 (ky=0): kx=0,1cos,1sin,2
grid[0][0] = take_first(pool0, lambda it: it["const_y"] and it["xmag"]==0, srt)
grid[0][1] = take_first(pool0, lambda it: it["const_y"] and it["xmag"]==1 and it["kind"]=="cos", srt)
grid[0][2] = take_first(pool0, lambda it: it["const_y"] and it["xmag"]==1 and it["kind"]=="sin", srt)
grid[0][3] = take_first(pool0, lambda it: it["const_y"] and it["xmag"]==2, srt)

# Rows 1 & 2 (|ky|=1): left col constant in x (kx=0), cos then sin
grid[1][0] = take_first(pool1, lambda it: it["const_x"] and it["kind"]=="cos", srt)
grid[2][0] = take_first(pool1, lambda it: it["const_x"] and it["kind"]=="sin", srt)
for c in [1,2,3]:
    grid[1][c] = take_first(pool1, lambda it: not it["const_x"], srt)
for c in [1,2,3]:
    grid[2][c] = take_first(pool1, lambda it: not it["const_x"], srt)

# Row 3 (|ky|=2): left col kx=0, then xmag=1 (cos,sin), then xmag=2
grid[3][0] = take_first(pool2, lambda it: it["const_x"], srt)
grid[3][1] = take_first(pool2, lambda it: it["xmag"]==1 and it["kind"]=="cos", srt)
grid[3][2] = take_first(pool2, lambda it: it["xmag"]==1 and it["kind"]=="sin", srt)
grid[3][3] = take_first(pool2, lambda it: it["xmag"]==2, srt)

ordered = [grid[r][c] for r in range(4) for c in range(4)]
assert all(it is not None for it in ordered)

# --- Orthonormality check -----------------------------------------------------
B = np.stack([it["fd"].ravel() for it in ordered], axis=1)
G = (B.T @ B) / (N*N)
print("Orthonormal (max off-diag):", float(np.max(np.abs(G - np.eye(16)))))

# --- Manual permutation hook --------------------------------------------------
PERM = list(range(16))  # edit this to re-order panels
ordered = [ordered[i] for i in PERM]

print("\nPanel index → label (before PERM):")
SELF = {(0,0),(2,0),(0,2),(2,2)}
for idx, it in enumerate([grid[r][c] for r in range(4) for c in range(4)]):
    k1,k2 = it["k"]
    tag = f"{'√2·' if (k1,k2) not in SELF else ''}{it['kind']}[{k1},{k2}]"
    #print(f"{idx:2d}: {tag}  (|kx|={it['xmag']}, |ky|={it['ymag']})")

# --- Helpers ------------------------------------------------------------------
def draw_base_grid(ax, N, z=0.0, major_step=1.0, minor_step=0.5):
    """Draw a 2D grid on plane z at integer coordinates (and optional minors)."""
    # Major lines
    vals = np.arange(0, N+1, major_step)
    for xi in vals:
        ax.plot([xi, xi], [0, N], [z, z], color=GRID_COLOR_MAJOR,
                linewidth=GRID_LW_MAJOR, alpha=GRID_ALPHA_MAJOR)
    for yi in vals:
        ax.plot([0, N], [yi, yi], [z, z], color=GRID_COLOR_MAJOR,
                linewidth=GRID_LW_MAJOR, alpha=GRID_ALPHA_MAJOR)
    # Minor lines
    if minor_step and minor_step > 0 and minor_step < major_step:
        vals_minor = np.arange(0, N+1, minor_step)
        # remove majors to avoid double-draw
        majors = set(np.round(vals, 8).tolist())
        for xi in vals_minor:
            if np.round(xi,8) in majors:
                continue
            ax.plot([xi, xi], [0, N], [z, z], color=GRID_COLOR_MINOR,
                    linewidth=GRID_LW_MINOR, alpha=GRID_ALPHA_MINOR)
        for yi in vals_minor:
            if np.round(yi,8) in majors:
                continue
            ax.plot([0, N], [yi, yi], [z, z], color=GRID_COLOR_MINOR,
                    linewidth=GRID_LW_MINOR, alpha=GRID_ALPHA_MINOR)

# --- Plot ---------------------------------------------------------------------
fig = plt.figure(figsize=(8,8))
for i, it in enumerate(ordered, start=1):
    ax = fig.add_subplot(4, 4, i, projection='3d')
    # smooth surface
    from matplotlib import cm
    from matplotlib.colors import Normalize

    norm = Normalize(vmin=it["fs"].min(), vmax=it["fs"].max())
    colors = cm.viridis(norm(it["fs"]))

    ax.plot_surface(X1s, X2s, it["fs"],
                facecolors=colors,
                linewidth=0, antialiased=True, alpha=0.8)
    # optional base plane (very light)
    if False:
        ax.plot_surface(X1s, X2s, np.full_like(X1s, BASE_Z), linewidth=0, alpha=0.08)
    # # discrete points as crosses
    # x = X1d.ravel()
    # y = X2d.ravel()
    # z = it["fd"].ravel()
    # ax.scatter(x, y, z, marker='x', s=50, depthshade=False, linewidths=0.9, color  = "grey")
    # # vertical drop lines to BASE_Z
    # for xi, yi, zi in zip(x, y, z):
    #     ax.plot([xi, xi], [yi, yi], [BASE_Z, zi], linewidth=2.0, alpha=0.9, color='grey')
    # base grid
    # if SHOW_BASE_GRID:
    #     draw_base_grid(ax, N, z=BASE_Z, major_step=GRID_MAJOR_STEP, minor_step=GRID_MINOR_STEP)
    # title
    k1,k2 = it["k"]
    #print(f"Panel {i-1:2d}: k=({k1},{k2}), kind={it['kind']}, |kx|={it['xmag']}, |ky|={it['ymag']}")
    eigenvalue = 4* (k1 > 0) + 4* (k2 > 0)  # Laplacian eigenvalue
    #title = f"{'√2·' if (k1,k2) not in SELF else ''}{it['kind']}[{k1},{k2}]   (|kx|={it['xmag']}, |ky|={it['ymag']})\nEigenvalue: {eigenvalue}"
    #title = f"{'√2·' if (k1,k2) not in SELF else ''}{it['kind']}[{k1},{k2}], $\lambda$ = {eigenvalue}"
    title = f"$\lambda$ = {eigenvalue}"
    ax.set_title(title, fontsize=9, y=0.99)  # smaller y brings it lower
    ax.set_xticks(range(N)); ax.set_yticks(range(N)); ax.set_zticks([-1, 0, 1])
    #ax.set_xlabel("x₁"); ax.set_ylabel("x₂")

        # Control tick label font size
    ax.tick_params(axis="both", which="major", labelsize=ticksize)
    ax.tick_params(axis="both", which="minor", labelsize=ticksize)
    ax.zaxis.set_tick_params(labelsize=ticksize)
#fig.suptitle("Real Orthonormal Fourier Basis (N=4)\nDiscrete 'X' samples + drop lines + smooth overlay + base grid", fontsize=14)
#fig.suptitle("Basis Vectors", fontsize=titlesize+3)
plt.tight_layout()
out_path = os.path.join(figures_dir, 'basis_vectors.pdf')
plt.savefig(out_path, dpi=dpi)
print(f'Saved → {out_path}')
