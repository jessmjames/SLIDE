# `raw_data/` — generated simulation products (provenance)

These `*.pkl` files are **generated, not tracked in git** (`.gitignore` excludes `*.pkl`). This
README *is* tracked, so the provenance is findable even when the data isn't in the repo.

**Regenerate everything (idempotent — skips any product whose pkl already exists):**

```bash
conda activate direvo            # jax 0.7.2, GPU
XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/generate_figure5_raw_data.py
```

Every pkl embeds its own `params` + `metadata` (incl. `paper_reference`) dict — inspect with
`pickle.load` if in doubt. Env knobs: `NK_GRID_LANDSCAPES`, `NK_GRID_BATCH`, `DE_STARTS`,
`DE_REPS`, `DE_BATCH`, `DE_GENERATIONS`, `HEATMAP_GEN`, `DE_POPSIZE`.

## Products

**Timings are single-run, per-product wall-clock on an RTX 5090 (WSL2).** ⓜ = measured this run
(from logs / consecutive file mtimes); ⓔ = earlier estimate, not freshly re-measured. The *total*
project time was several hours — but that is cumulative across many re-run / debug cycles and idle
gaps between runs, **not** the cost of generating any single product (and not the sum below).

| File | Figure | Contents / shape | Key params | Gen time |
|------|--------|------------------|------------|----------|
| `nk_strategy_grid_raw_data.pkl` | **Fig 5A** | NK strategy look-up, 100 (N,K) pts × (split,base,reps) | N 10–50, **200 landscapes**, 25 reps, 7×7 grid, M=25, pop 1200, mut 0.1/N | ⓔ ~78 min |
| `nk_strategy_grid_M50_raw_data.pkl` | **Fig S5** | same, M=50 | as above but **50 landscapes**, M=50 | ⓜ 23.5 min |
| `nk_strategy_grid_M100_raw_data.pkl` | **Fig S5** | same, M=100 | as above but **50 landscapes**, M=100 | ⓜ 44 min |
| `nk_decay_N4_A20_raw_data.pkl` | Fig 5 B/C | NK decay curves, N=4 A=20, K∈{1,2,3} | 125 landscapes, ≤10k starts | ⓔ ~2 min |
| `nk_strategy_N4_A20_raw_data.pkl` | Fig 5 B/C | NK strategy lookup, N=4 A=20 | 7×7 grid | ⓜ ~5 min |
| `nk_decay_N3_A20_raw_data.pkl` | Fig 5 (ParD3) | NK decay curves, N=3 A=20, K∈{1,2} | 125 landscapes | ⓔ ~2 min |
| `nk_strategy_N3_A20_raw_data.pkl` | Fig 5 (ParD3) | NK strategy lookup, N=3 A=20 | 5×5 grid | ⓜ ~2.5 min |
| `empirical_decay_{GB1,TrpB,TEV,ParD3}_uniform_raw_data.pkl` | Fig 5 D–G | empirical fitness-decay curves | ≤10k starts, uniform mutation | ⓜ ~1.7 min each |
| `empirical_strategy_{GB1,TrpB,TEV,ParD3}_uniform_raw_data.pkl` | Fig 5 D–G | empirical strategy sweep | 7×7 (5×5 ParD3) | ⓜ ~1 min each |
| `empirical_strategy_traj_{GB1,TrpB,TEV,ParD3}_raw_data.pkl` | **Fig 5 D–G** | **best-variant trajectory of every (split,base) cell** — heatmap = gen-25 slice, DE lines = selected cells (ONE dataset) | 100 starts × 300 reps, 150 gens, pop 1200 (60 ParD3), mut 0.1/N, best-variant max | ⓜ ~4–6.5 min each |

## Calibration notes (so the figures stay consistent)

- **Heatmap = gen-25 slice** of the trajectory sweep (`HEATMAP_GEN=25`); DE lines = the full
  150-gen trajectory of the same cells → `line[gen25] == heatmap cell` by construction.
- ParD3 is **N=3** (5×5 grid, popsize 60). Other landscapes N=4, popsize 1200.
- NK strategy grid is **execution-bound** at high M / many landscapes (M=100 × 200 landscapes ≈
  hours); `k`/`mutation_rate` are traced (not JIT-static) so the kernel compiles once per N.
  This optimisation is **verified bit-exact** (max|Δ|=0) vs the old static-`k` kernel on the
  worst-case point (N=50, K=50, 200 landscapes) — i.e. `nk_strategy_grid` (old code) and the M50/M100
  grids (new code) are directly comparable.

## Archives (kept for reference — alternative parameterisations)

- `archive_panelA_200landscapes/` — backup of the 200-landscape Fig 5A grid.
- `archive_150gen_m0.01/` — long-horizon empirical sweeps at mut 0.01, 150 gens (the
  earlier parameterisation before anchoring to the gen-25 heatmap calibration).
