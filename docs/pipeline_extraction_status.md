# Pipeline Extraction Status

Tracks the **actual implementation state** of each figure's 3-file pipeline.
✓ = fully implemented and runnable. Stub = script exists but does nothing useful.

Last audited: 2026-03-31.

---

## Tier 1 — Fully runnable end-to-end

All three scripts work. Running `1_raw_data.py → 2_process_data.py → 3_plot.py` produces the figure.

| Figure | Notes |
|--------|-------|
| 2A–2E | Self-contained; no external data needed. |
| 4A | Reads static landscape arrays only. `2_process_data.py` implemented. |
| 4B | `2_process_data.py` implemented. ⚠️ Filename mismatch: `landscape_heterogeneity.py` writes `N4A20_heterogeneity2.pkl` but `2_process_data.py` may expect `N4A20_heterogeneity.pkl` — needs checking. |
| 4D | `2_process_data.py` implemented; shares `decay_curves_*_all_starts.pkl` with 4B. |
| S2 | Complete inline implementation. `plot_data/` pkls exist from Feb 2026. |
| S3 | All 3 scripts work; 2 and 3 delegate to `scripts/` wrappers (all on main). |
| Fig 6 | All 3 scripts work; all delegate to `scripts/` wrappers (all on main). |

---

## Tier 2 — `3_plot.py` works; `2_process_data.py` is a stub

Simulation scripts exist on main. Just need to extract the data-processing notebook cells.

| Figure | What `2_process_data.py` must do | Source | SLIDE_data input |
|--------|----------------------------------|--------|-----------------|
| **3A** | Run DE inline (N=20 K=14 and K=1); fit decay rates; save `smooth_rugged_example.pkl` | `ruggedness_figures_data_processing.ipynb` cells 8–9 | None (self-contained) |
| **3B** | Load sweep; fit decay rates per NK pair; save `ruggedness_accuracy.pkl` | data_processing.ipynb cells 5–14 | `SLIDE_data/large_decay_curve_sweep.pkl` ← `scripts/large_decay_curve_sweep.py` ✓ |
| **3C** | Load sweep; fit decay rates per popsize; save `popsize_accuracy.pkl` | data_processing.ipynb cells 16–19 | `SLIDE_data/popsize_accuracy.pkl` ← `scripts/popsize_accuracy.py` ✓ |
| **3D** | Load sweep; fit decay rates per mut rate; save `mut_accuracy.pkl` | data_processing.ipynb cells 21–24 | `SLIDE_data/mutation_rate_accuracy.pkl` ← `scripts/mutation_rate_accuracy.py` ✓ |
| **3E** | Build N=12 NK landscapes analytically; compute 7 ruggedness metrics; save `NK_ruggedness_metric_comparison.pkl` | data_processing.ipynb cells 26–31 | None (self-contained, ~50 landscapes × 12 K values) |
| **4C** | Load heterogeneity data; compute per-repeat rho values | `ruggedness_figures_data_processing_IK.ipynb` cells 48–50 | Shares `decay_curves_*_all_starts.pkl` with 4B |
| **4E** | Load popsize decay curves; bootstrap rho at each popsize; save `estimation_variance.pkl` (array_results1) | data_processing.ipynb (long computation) | `SLIDE_data/decay_curves_*_multi_popsize.pkl` ← `scripts/empirical_landscape_decay_curves_popsize.py` ✓ |
| **4F** | Same as 4E (array_results2 — bootstrap over generations); saves same pkl | Same cells | Same |

---

## Tier 3 — `3_plot.py` works; `2_process_data.py` is a stub with complex extraction needed

These figures require extracting multi-cell data-processing logic from `ruggedness_figures_data_processing.ipynb`. Simulation scripts all exist on main.

| Figure | What `2_process_data.py` must do | Source cells | Key SLIDE_data inputs |
|--------|----------------------------------|--------------|----------------------|
| **5B / 5C** | Derive NK strategy predictions; run DE simulations (100 reps × 4 conditions); save `NK_DE.pkl`, `NK_strategy_spaces.pkl` | data_processing.ipynb cells 98–104 | `large_strategy_sweep.pkl`, `large_decay_curve_sweep.pkl` |
| **5D** | Compute empirical lookup; run GB1 strategy selection + DE test; save `GB1_strategy_selection.pkl` | cells 106–113 | `N4A20_strategy_sweep.pkl`, `N4A20_decay_curves.pkl`, `decay_curves_gb1_m0.1_multistart_10000_uniform.pkl`, `strategy_sweep_GB1_multistart_100_uniform_m0.025.pkl` |
| **5E** | Same as 5D for TrpB; save `TrpB_strategy_selection.pkl` | cells 106–109, 114 | Same N4A20 + TrpB curve pkls |
| **5F** | Same as 5D for TEV; save `TEV_strategy_selection.pkl` | cells 106–109, 115 | Same N4A20 + TEV curve pkls |
| **5G** | Compute N=3 lookup; run ParD3 strategy selection + DE test; save `ParD3_strategy_selection.pkl` | cells 116–119 | `N3A20_strategy_sweep.pkl`, `N3A20_decay_curves.pkl`, `decay_curves_pard3_m0.1_multistart_10000_uniform.pkl`, `strategy_sweep_E3_multistart_100_uniform_m0.025.pkl` |

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
