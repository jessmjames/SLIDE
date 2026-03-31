# Pipeline Extraction Status

Tracks the **actual implementation state** of each figure's 3-file pipeline.
✓ = fully implemented and runnable. Stub = script exists but does nothing useful.

Last audited: 2026-03-31. Last updated: 2026-03-31 (5B–5G extracted).

---

## Tier 1 — Fully runnable end-to-end

All three scripts work. Running `1_raw_data.py → 2_process_data.py → 3_plot.py` produces the figure.

| Figure | Notes |
|--------|-------|
| 2A–2E | Self-contained; no external data needed. |
| **3A** | `2_process_data.py` extracted 2026-03-31. Runs DE inline; no SLIDE_data needed. |
| **3B** | `2_process_data.py` extracted 2026-03-31. Reads `large_decay_curve_sweep.pkl`. |
| **3C** | `2_process_data.py` extracted 2026-03-31. Reads `popsize_accuracy.pkl`. |
| **3D** | `2_process_data.py` extracted 2026-03-31. Reads `mutation_rate_accuracy.pkl`. |
| **3E** | `2_process_data.py` extracted 2026-03-31. Self-contained; builds NK landscapes analytically (~minutes). |
| 4A | Reads static landscape arrays only. `2_process_data.py` implemented. |
| 4B | `2_process_data.py` implemented. ⚠️ Filename mismatch: `landscape_heterogeneity.py` writes `N4A20_heterogeneity2.pkl` but `2_process_data.py` expects `N4A20_heterogeneity.pkl` — needs checking. |
| **4C** | `2_process_data.py` extracted 2026-03-31. Delegates to 4B (shared pkl). |
| 4D | `2_process_data.py` implemented; shares `decay_curves_*_all_starts.pkl` with 4B. |
| **4E** | `2_process_data.py` extracted 2026-03-31. Bootstrap rho over popsizes; reads `*_multi_popsize.pkl`. |
| **4F** | `2_process_data.py` extracted 2026-03-31. Delegates to 4E (shared pkl). |
| S2 | Complete inline implementation. `plot_data/` pkls exist from Feb 2026. |
| S3 | All 3 scripts work; 2 and 3 delegate to `scripts/` wrappers (all on main). |
| **5B** | `2_process_data.py` extracted 2026-03-31. Runs NK DE (4 conditions, 100 reps each); reads `large_strategy_sweep.pkl`, `large_decay_curve_sweep.pkl`. |
| **5C** | `2_process_data.py` extracted 2026-03-31. Delegates to 5B (shared pkls). |
| **5D** | `2_process_data.py` extracted 2026-03-31. Builds N4A20 lookup + runs GB1 DE; ⚠️ `strategy_sweep_GB1_multistart_100_uniform_m0.025.pkl` has no generation script on main. |
| **5E** | `2_process_data.py` extracted 2026-03-31. TrpB DE; shares N4A20 lookup; ⚠️ same warning for TrpB sweep pkl. |
| **5F** | `2_process_data.py` extracted 2026-03-31. TEV DE; shares N4A20 lookup; ⚠️ same warning for TEV sweep pkl. |
| **5G** | `2_process_data.py` extracted 2026-03-31. N3A20 lookup + ParD3 DE; ⚠️ `strategy_sweep_E3_multistart_100_uniform_m0.025.pkl` has no generation script on main. |
| Fig 6 | All 3 scripts work; all delegate to `scripts/` wrappers (all on main). |

---

## Tier 2 — `3_plot.py` works; `2_process_data.py` is a stub

*(Was populated 2026-03-31 — all figures moved to Tier 1 above.)*

| Figure | What `2_process_data.py` must do | Source | SLIDE_data input |
|--------|----------------------------------|--------|-----------------|
| **3A** | Run DE inline (N=20 K=14 and K=1); fit decay rates; save `smooth_rugged_example.pkl` | `ruggedness_figures_data_processing.ipynb` cells 8–9 | None (self-contained) |
| **3B** | Load sweep; fit decay rates per NK pair; save `ruggedness_accuracy.pkl` | data_processing.ipynb cells 5–14 | `SLIDE_data/large_decay_curve_sweep.pkl` ← `scripts/large_decay_curve_sweep.py` ✓ |
| **3C** | Load sweep; fit decay rates per popsize; save `popsize_accuracy.pkl` | data_processing.ipynb cells 16–19 | `SLIDE_data/popsize_accuracy.pkl` ← `scripts/popsize_accuracy.py` ✓ |
| **3D** | Load sweep; fit decay rates per mut rate; save `mut_accuracy.pkl` | data_processing.ipynb cells 21–24 | `SLIDE_data/mutation_rate_accuracy.pkl` ← `scripts/mutation_rate_accuracy.py` ✓ |
| ~~3E~~ | ✓ Done 2026-03-31 | — | — |
| ~~4C~~ | ✓ Done 2026-03-31 | — | — |
| ~~4E~~ | ✓ Done 2026-03-31 | — | — |
| ~~4F~~ | ✓ Done 2026-03-31 | — | — |

---

## Tier 3 — `3_plot.py` works; `2_process_data.py` is a stub with complex extraction needed

*(Was populated 2026-03-31 — all figures moved to Tier 1 above.)*

| Figure | What `2_process_data.py` must do | Source cells | Key SLIDE_data inputs |
|--------|----------------------------------|--------------|----------------------|
| ~~5B / 5C~~ | ✓ Done 2026-03-31 | — | — |
| ~~5D~~ | ✓ Done 2026-03-31 | — | — |
| ~~5E~~ | ✓ Done 2026-03-31 | — | — |
| ~~5F~~ | ✓ Done 2026-03-31 | — | — |
| ~~5G~~ | ✓ Done 2026-03-31 | — | — |

Note: Fig 5A is handled — `2_process_data.py` calls `scripts/plot_strategy_prediction.py` which does the full processing.

---

## Tier 4 — Major gaps

| Figure | Problem | What's needed |
|--------|---------|---------------|
| **Fig 7** | `fourier_analysis.pkl` has no reproducible generation script. `1_raw_data.py` is a docstring stub. | Write a proper `1_raw_data.py` that builds NK landscapes for N=10, A=20, Ks=[0,1,3,5,9] using `get_nk_l_o_shape`. ⚠️ Memory-intensive: 20^10 ≈ 10^13 entries — may need to reduce parameters or confirm originals with Jess. |
| **S1** | Nothing on main. No `figures_src/` folder. 6 simulation scripts only on `rebuttals` branch. | Merge 6 scripts from `rebuttals`; create `figures_src/figure_S1_alt_landscapes/` with 6 `1_raw_data_*.py` files + stub `2_process_data.py` + `3_plot.py`. |
| **S4** | Source disputed — awaiting Jess. Current `3_plot.py` likely wrong. | Confirm with Jess then extract. |
| **4G** | All data lives on Jess's machine. `1_raw_data.py` / `2_process_data.py` are stubs noting Jess's paths. | Copy 5 pkl files from Jess's machine to `plot_data/`. No simulation extraction needed — data can't be regenerated without Jess. |

---

## Shared simulation data

Running one script covers multiple figures — important for prioritisation:

| Script (all on main) | Generates | Needed by |
|---------------------|-----------|-----------|
| `scripts/large_decay_curve_sweep.py` | `large_decay_curve_sweep.pkl` | 3B, 5A, 5B/5C |
| `scripts/large_strategy_sweep.py` | `large_strategy_sweep.pkl` | 5A, 5B/5C |
| `scripts/empirical_landscape_decay_curves_all_starts.py` | `decay_curves_*_all_starts.pkl` (4 landscapes) | 4B, 4C, 4D |
| `scripts/empirical_landscape_decay_curves_popsize.py` | `decay_curves_*_multi_popsize.pkl` (4 landscapes) | 4E, 4F |
| `scripts/strategy_sweep_N4A20.py` | `N4A20_strategy_sweep.pkl` | 5D, 5E, 5F |
| `scripts/decay_curves_N4A20.py` | `N4A20_decay_curves.pkl` | 5D, 5E, 5F |
| `scripts/strategy_sweep_N3A20.py` | `N3A20_strategy_sweep.pkl` | 5G |
| `scripts/decay_curves_N3A20.py` | `N3A20_decay_curves.pkl` | 5G |
| `scripts/empirical_landscape_decay_curves_codon_fast_75steps.py` | nuc codon decay curves | S3, Fig 6 |
| `scripts/landscape_heterogeneity.py` | `N4A20_heterogeneity.pkl` | 4B, 4C |
