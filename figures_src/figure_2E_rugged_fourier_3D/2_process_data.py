"""
Figure 2E — Step 2: process data.

Constructs the real orthonormal Fourier basis for N=4, defines bump_func,
and projects it onto the basis to obtain Fourier coefficients C_bump (4×4).

    bump_func(x, y) = -sin(x*pi/2 + y*pi/2) * cos(y*pi/4)

No pkl output — computed inline and used directly by 3_plot.py.

Originally from ruggedness_figures_plots.ipynb cells 84, 85, 88, 89, 91.
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

def bump_func(x, y):
    return -xp.sin(x * xp.pi / 2 + y * xp.pi / 2) * xp.cos(y * xp.pi / 4)

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

def project_to_fourier_coeffs(f_d, ordered, N):
    fvec = np.asarray(f_d, dtype=float).ravel()
    B = np.stack([it["fd"].ravel() for it in ordered], axis=1)
    coeffs = (B.T @ fvec) / (N * N)
    C = coeffs.reshape(4, 4)
    return C

bump_func_d = np.array(bump_func(X1d, X2d))
C_bump = project_to_fourier_coeffs(bump_func_d, ordered, N)

if __name__ == "__main__":
    print(f"C_bump:\n{C_bump}")
    print("No pkl files written — arrays are computed inline.")
