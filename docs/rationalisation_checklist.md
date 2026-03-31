# Rationalisation Checklist

Goal: every figure gets a folder in `figures_src/` with three files:
- `1_raw_data.py` — runs simulations → `SLIDE_data/`
- `2_process_data.py` — `SLIDE_data/` → `plot_data/*.pkl`
- `3_plot.py` — `plot_data/*.pkl` → `figures/*.pdf`

See `docs/figure_pipelines.md` for full pipeline details per figure.

---

## Quick fixes (do first)

- [x] **Fig 6** — change `savefig` output in `scripts/plot_mutation_model_comparison.py` from `accuracy_over_sampling_nuc_models.pdf` to `graph_mutation_model.pdf`
- ⚠️ **S4** — source unclear; see note in `docs/figure_pipelines.md`. Awaiting confirmation from Jess.
- [x] **4C** — violin plot extracted into `figures_src/figure_4C_repeat_heterogeneity/3_plot.py` (scripts, not notebook)

---

## Phase 1: figures with existing standalone scripts

These are closest to the target structure — just need organising into folders.

- [x] **S2** `rho_accuracy_mutation_models.pdf` — `figures_src/figure_S2_rho_accuracy/`
- [x] **S3** `decay_curves_nuc_75steps.pdf` — `figures_src/figure_S3_decay_curves_nuc/`
- [x] **Fig 6** `graph_mutation_model.pdf` — `figures_src/figure_6_mutation_model/`
- [x] **Fig 4C** `repeat_heterogeneity.pdf` — `figures_src/figure_4C_repeat_heterogeneity/`
- ⚠️ **S4** `strategy_prediction_{g}gens.pdf` — source disputed; see `docs/figure_pipelines.md`. Current `figures_src/figure_S4_strategy_gens/3_plot.py` outputs heatmaps which may be wrong.

---

## Phase 2: notebook-only figures (extract into scripts)

For each figure: identify the relevant notebook cells, trace data dependencies,
write the three scripts.

- [x] **Fig 2A** `smooth_landscape_3D.pdf` — `figures_src/figure_2A_smooth_landscape_3D/` ✓ verified
- [x] **Fig 2B** `rugged_landscape_3D.pdf` — `figures_src/figure_2B_rugged_landscape_3D/` ✓ verified
- [x] **Fig 2C** `basis_vectors.pdf` — `figures_src/figure_2C_basis_vectors/` ✓ verified
- [x] **Fig 2D** `smooth_fourier_3D.pdf` — `figures_src/figure_2D_smooth_fourier_3D/` ✓ verified
- [x] **Fig 2E** `rugged_fourier_3D.pdf` — `figures_src/figure_2E_rugged_fourier_3D/` ✓ verified
- [x] **Fig 3A** `decay_curves_example.pdf` — `figures_src/figure_3A_decay_curves_example/` ✓ verified
- [x] **Fig 3B** `accuracy_over_K.pdf` — `figures_src/figure_3B_accuracy_over_K/` ✓ verified
- [x] **Fig 3C** `accuracy_over_popsize.pdf` — `figures_src/figure_3C_accuracy_over_popsize/` ✓ verified
- [x] **Fig 3D** `accuracy_over_mut.pdf` — `figures_src/figure_3D_accuracy_over_mut/` ✓ verified
- [x] **Fig 3E** `NK_ruggedness_metric_comparison.pdf` — `figures_src/figure_3E_NK_ruggedness_comparison/` ✓ verified
- [x] **Fig 4A** `empirical_fourier_spectra.pdf` — `figures_src/figure_4A_empirical_fourier_spectra/` ✓ verified
- [x] **Fig 4B** `landscape_heterogeneity.pdf` — `figures_src/figure_4B_landscape_heterogeneity/` ✓ verified
- [x] **Fig 4D** `accuracy_over_sampling.pdf` — `figures_src/figure_4D_accuracy_over_sampling/` ✓ verified
- [x] **Fig 4E** `empirical_variance_over_popsize.pdf` — `figures_src/figure_4E_empirical_variance_popsize/` ✓ verified
- [x] **Fig 4F** `empirical_variance_over_generations.pdf` — `figures_src/figure_4F_empirical_variance_generations/` ✓ verified
- [x] **Fig 4C** (repeat heterogeneity — violin) — `figures_src/figure_4C_repeat_heterogeneity/`
- [x] **Fig 4G** `empirical_ruggedness_metric_comparison_IK.pdf` — `figures_src/figure_4G_empirical_ruggedness_IK/` ✓ verified (⚠️ data pkls from Jess's machine — copy to `plot_data/` before running)
- [x] **Fig 5A** `strategy_prediction.pdf` — `figures_src/figure_5A_strategy_prediction/` ✓ verified
- [x] **Fig 5B** `N45K1_DE_*.pdf` — `figures_src/figure_5B_NK_DE/` ✓ verified
- [x] **Fig 5C** `N45K25_DE_*.pdf` — `figures_src/figure_5C_NK_rugged_DE/` ✓ verified
- [x] **Fig 5D** `GB1_*.pdf` — `figures_src/figure_5D_GB1_DE/` ✓ verified
- [x] **Fig 5E** `TrpB_*.pdf` — `figures_src/figure_5E_TrpB_DE/` ✓ verified
- [x] **Fig 5F** `TEV_*.pdf` — `figures_src/figure_5F_TEV_DE/` ✓ verified (⚠️ title says "TrpB" — copy-paste error in original notebook, faithfully preserved)
- [x] **Fig 5G** `ParD3_*.pdf` — `figures_src/figure_5G_ParD3_DE/` ✓ verified (cell 77 copy-paste error intentionally excluded)
- [x] **Fig 7** `NK_spectra.pdf` — `figures_src/figure_7_NK_spectra/` ✓ verified (`np.concat` is valid in NumPy ≥ 2.0)
- ⚠️ **S4** (source unclear — awaiting Jess) — `figures_src/figure_S4_strategy_gens/` exists but plotting may be wrong

---

## Phase 3: figures with scattered/incomplete code

- [ ] **S1** `alt_landscapes.pdf`
  - [ ] Merge `decay_curve_sweep_alt_landscapes_*.py` and `strategy_sweep_alt_landscapes_*.py` from `rebuttals` into main
  - [ ] Extract plotting cell from `ruggedness_figures_data_processing.ipynb` (`rebuttals`)
  - [ ] Create full 3-file structure in `figures_src/figure_S1_alt_landscapes/`
